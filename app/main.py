import asyncio
import logging
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from secrets import token_urlsafe
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from app.config import Settings, get_settings
from app.db import Database
from app.runtime import ProcessingGate, RequestControlMiddleware
from app.schemas import AnalysisResponse, Product, PublicResult, ShareLinkResponse, TryOnRequest, TryOnResponse
from app.services.color_analysis import ImageValidationError, analyze
from app.services.qr import make_qr_png
from app.services.storage import Storage
from app.services.virtual_tryon import provider_for

logger = logging.getLogger("uvicorn.error")

def services(settings: Settings = Depends(get_settings)) -> tuple[Database, Storage]:
    return Database(settings), Storage(settings)


def require_admin(x_admin_token: str | None = Header(default=None), settings: Settings = Depends(get_settings)) -> None:
    if not settings.admin_token or x_admin_token != settings.admin_token:
        raise HTTPException(404, "Not found")


def _analysis_response(row: dict, include_image: bool = True) -> AnalysisResponse:
    return AnalysisResponse(
        id=row["id"], status=row["status"], created_at=row["created_at"], expires_at=row["expires_at"],
        result=row.get("result_json"),
        original_image_url=f"/api/v1/analyses/{row['id']}/image" if include_image else None,
        error=row.get("error"),
    )


def _try_on_response(row: dict, analysis: dict | None = None) -> TryOnResponse:
    return TryOnResponse(
        id=row["id"], analysis_id=row["analysis_id"], status=row["status"], provider=row["provider"], created_at=row["created_at"],
        result_image_url=f"/api/v1/try-ons/{row['id']}/image" if row.get("result_path") else None,
        before_image_url=f"/api/v1/analyses/{row['analysis_id']}/image" if analysis else None,
        error=row.get("error"),
    )


def _rgb(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def matched_products(db: Database, result: dict) -> list[dict]:
    recommended = {item["category"]: _rgb(item["color"]) for item in result["recommendations"]}
    products = db.list_products(season=result["season"])
    for product in products:
        target, actual = recommended[product["category"]], _rgb(product["color_hex"])
        distance = sum((a - b) ** 2 for a, b in zip(target, actual)) ** 0.5
        product["match_score"] = max(0, round(100 - distance / 441.67 * 100))
    return sorted(products, key=lambda item: (-item["match_score"], item["category"], item["name"]))


def cleanup_expired(db: Database, storage: Storage) -> int:
    count = 0
    for analysis_id in db.expired_ids():
        storage.delete(db.delete_analysis(analysis_id))
        count += 1
    db.cleanup_shares()
    return count


async def periodic_cleanup(db: Database, storage: Storage, interval_minutes: int) -> None:
    while True:
        await asyncio.sleep(max(1, interval_minutes * 60))
        cleanup_expired(db, storage)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    db, storage = Database(settings), Storage(settings)
    db.init()
    cleanup_expired(db, storage)
    app.state.try_on_gate = ProcessingGate(settings.max_concurrent_try_ons)
    cleanup_task = asyncio.create_task(periodic_cleanup(db, storage, settings.cleanup_interval_minutes))
    yield
    cleanup_task.cancel()
    with suppress(asyncio.CancelledError):
        await cleanup_task


app = FastAPI(
    title="예뻐질 사람? It's Me! API",
    version="1.0.0",
    description="퍼스널컬러 분석, 메이크업 추천, 가상 메이크업 체험 백엔드",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)
app.add_middleware(RequestControlMiddleware, limit_per_minute=get_settings().rate_limit_per_minute, log_path=get_settings().log_path)


@app.get("/health")
def health(request: Request, settings: Settings = Depends(get_settings)):
    queue = request.app.state.try_on_gate.snapshot() if hasattr(request.app.state, "try_on_gate") else {"active": 0, "waiting": 0}
    return {"status": "ok", "vto_provider": settings.vto_provider, "try_on_queue": queue}


@app.post("/api/v1/analyses", response_model=AnalysisResponse, status_code=201)
async def create_analysis(
    image: UploadFile = File(..., description="정면 얼굴 JPG/PNG"),
    consent: bool = Form(..., description="얼굴 이미지 수집·처리 동의"),
    svc: tuple[Database, Storage] = Depends(services),
):
    if not consent:
        raise HTTPException(400, "얼굴 이미지 수집·처리에 동의해야 체험할 수 있습니다.")
    db, storage = svc
    cleanup_expired(db, storage)
    try:
        path = await storage.save_upload(image)
    except ValueError as exc:
        raise HTTPException(413 if "MB" in str(exc) else 415, str(exc)) from exc

    analysis_id = uuid4().hex
    db.create_analysis(analysis_id, str(path))
    try:
        result = analyze(path)
        db.finish_analysis(analysis_id, result)
    except ImageValidationError as exc:
        db.fail_analysis(analysis_id, str(exc))
        storage.delete([str(path)])
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        logger.exception("analysis failed: %s", analysis_id)
        db.fail_analysis(analysis_id, "분석 중 오류가 발생했습니다.")
        storage.delete([str(path)])
        raise HTTPException(500, "분석 중 오류가 발생했습니다.") from exc
    return _analysis_response(db.get_analysis(analysis_id))


@app.get("/api/v1/analyses/{analysis_id}", response_model=AnalysisResponse)
def get_analysis(analysis_id: str, svc: tuple[Database, Storage] = Depends(services)):
    db, storage = svc
    cleanup_expired(db, storage)
    row = db.get_analysis(analysis_id)
    if not row:
        raise HTTPException(404, "분석 결과를 찾을 수 없거나 보관 기간이 만료되었습니다.")
    return _analysis_response(row)


@app.get("/api/v1/analyses/{analysis_id}/image", include_in_schema=False)
def get_original_image(analysis_id: str, svc: tuple[Database, Storage] = Depends(services)):
    db, storage = svc
    cleanup_expired(db, storage)
    row = db.get_analysis(analysis_id)
    path = storage.safe_file(row.get("original_path") if row else None, storage.uploads)
    if not path:
        raise HTTPException(404, "이미지를 찾을 수 없습니다.")
    return FileResponse(path, headers={"Cache-Control": "no-store"})


@app.post("/api/v1/try-ons", response_model=TryOnResponse, status_code=201)
async def create_try_on(payload: TryOnRequest, request: Request, settings: Settings = Depends(get_settings), svc: tuple[Database, Storage] = Depends(services)):
    db, storage = svc
    cleanup_expired(db, storage)
    analysis = db.get_analysis(payload.analysis_id)
    if not analysis or analysis["status"] != "completed":
        raise HTTPException(404, "완료된 분석 결과를 찾을 수 없습니다.")
    result = analysis["result_json"]
    defaults = {item["category"]: item["color"] for item in result["recommendations"]}
    colors = {
        "lip": payload.lip_color or defaults["lip"],
        "blush": payload.blush_color or defaults["blush"],
        "eye": payload.eye_color or defaults["eye"],
    }
    try_on_id = uuid4().hex
    now = datetime.now(UTC).isoformat()
    row = {"id": try_on_id, "analysis_id": payload.analysis_id, "created_at": now, "status": "processing", "provider": settings.vto_provider, "request": {"colors": colors, "intensity": payload.intensity}}
    admin_token = request.headers.get("X-Admin-Token")

    is_admin = bool(
        settings.admin_token
        and admin_token == settings.admin_token
    )

    limit_error = db.create_try_on(
        row,
        None if is_admin else settings.max_try_ons_per_analysis,
        settings.max_total_try_ons,
    )

    if limit_error == "analysis_limit":
        raise HTTPException(
            429, f"가상 메이크업은 최대 {settings.max_try_ons_per_analysis}회까지 체험할 수 있습니다."
        )

    if limit_error == "total_limit":
        raise HTTPException(
            429, "행사 전체 가상 메이크업 사용량이 초과되었습니다."
        )
    output = storage.results / f"{try_on_id}.jpg"
    try:
        async with request.app.state.try_on_gate:
            provider = provider_for(settings)
            task_id = await provider.render(Path(analysis["original_path"]), output, colors, payload.intensity, result["face_box"])
            db.finish_try_on(try_on_id, str(output), task_id)
    except Exception as exc:
        logger.exception("try-on failed: %s", try_on_id)
        output.unlink(missing_ok=True)
        db.fail_try_on(try_on_id, str(exc))
        raise HTTPException(502, "가상 메이크업 처리에 실패했습니다. 잠시 후 다시 시도해 주세요.") from exc
    return _try_on_response(db.get_try_on(try_on_id), analysis)

@app.get("/api/v1/analyses/{analysis_id}/try-on-usage")
def get_try_on_usage(
    analysis_id: str,
    request: Request,
    settings: Settings = Depends(get_settings),
    svc: tuple[Database, Storage] = Depends(services),
):
    from fastapi.responses import JSONResponse

    db, storage = svc
    cleanup_expired(db, storage)

    analysis = db.get_analysis(analysis_id)

    if not analysis or analysis["status"] != "completed":
        raise HTTPException(404, "분석 결과를 찾을 수 없습니다.")

    admin_token = request.headers.get("X-Admin-Token")

    is_admin = bool(
        settings.admin_token
        and admin_token == settings.admin_token
    )

    usage = db.get_try_on_usage(analysis_id)

    used = usage["used"]

    total_remaining = max(
        0, settings.max_total_try_ons - usage["total_used"]
    )

    limit = None if is_admin else settings.max_try_ons_per_analysis

    remaining = (
        total_remaining
        if is_admin
        else min(max(0, limit - used), total_remaining)
    )

    return JSONResponse(
        content={
            "is_admin": is_admin,
            "limit": limit,
            "used": used,
            "remaining": remaining,
        },
        headers={"Cache-Control": "no-store"},
    )

@app.get("/api/v1/try-ons/{try_on_id}", response_model=TryOnResponse)
def get_try_on(try_on_id: str, svc: tuple[Database, Storage] = Depends(services)):
    db, storage = svc
    cleanup_expired(db, storage)
    row = db.get_try_on(try_on_id)
    if not row:
        raise HTTPException(404, "가상 메이크업 결과를 찾을 수 없습니다.")
    return _try_on_response(row, db.get_analysis(row["analysis_id"]))


@app.get("/api/v1/try-ons/{try_on_id}/image", include_in_schema=False)
def get_try_on_image(try_on_id: str, svc: tuple[Database, Storage] = Depends(services)):
    db, storage = svc
    cleanup_expired(db, storage)
    row = db.get_try_on(try_on_id)
    path = storage.safe_file(row.get("result_path") if row else None, storage.results)
    if not path:
        raise HTTPException(404, "결과 이미지를 찾을 수 없습니다.")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.delete("/api/v1/analyses/{analysis_id}", status_code=204)
def delete_analysis(analysis_id: str, svc: tuple[Database, Storage] = Depends(services)):
    db, storage = svc
    if not db.get_analysis(analysis_id):
        raise HTTPException(404, "분석 결과를 찾을 수 없습니다.")
    storage.delete(db.delete_analysis(analysis_id))


@app.get("/api/v1/products", response_model=list[Product])
def list_products(season: str | None = None, category: str | None = None, svc: tuple[Database, Storage] = Depends(services)):
    if season and season not in {"spring", "summer", "autumn", "winter"}:
        raise HTTPException(422, "올바르지 않은 퍼스널컬러 유형입니다.")
    if category and category not in {"lip", "blush", "eye"}:
        raise HTTPException(422, "올바르지 않은 제품 분류입니다.")
    return svc[0].list_products(season, category)


@app.get("/api/v1/analyses/{analysis_id}/products", response_model=list[Product])
def recommended_products(analysis_id: str, svc: tuple[Database, Storage] = Depends(services)):
    db, storage = svc
    cleanup_expired(db, storage)
    analysis = db.get_analysis(analysis_id)
    if not analysis or not analysis.get("result_json"):
        raise HTTPException(404, "분석 결과를 찾을 수 없습니다.")
    return matched_products(db, analysis["result_json"])


@app.put("/api/v1/admin/products/{product_id}", response_model=Product, dependencies=[Depends(require_admin)])
def save_product(product_id: str, product: Product, svc: tuple[Database, Storage] = Depends(services)):
    if product_id != product.id:
        raise HTTPException(400, "경로와 제품 ID가 일치하지 않습니다.")
    svc[0].upsert_product(product.model_dump())
    return product


@app.post("/api/v1/analyses/{analysis_id}/share", response_model=ShareLinkResponse, status_code=201)
def create_share_link(analysis_id: str, settings: Settings = Depends(get_settings), svc: tuple[Database, Storage] = Depends(services)):
    db, storage = svc
    cleanup_expired(db, storage)
    analysis = db.get_analysis(analysis_id)
    if not analysis or analysis["status"] != "completed":
        raise HTTPException(404, "완료된 분석 결과를 찾을 수 없습니다.")
    token = token_urlsafe(24)
    share = db.create_share(token, analysis_id, settings.share_ttl_minutes)
    base = settings.public_base_url.rstrip("/")
    return {"token": token, "expires_at": share["expires_at"], "result_url": f"{base}/api/v1/public/results/{token}", "qr_image_url": f"{base}/api/v1/public/results/{token}/qr"}


@app.get("/api/v1/public/results/{token}", response_model=PublicResult)
def public_result(token: str, svc: tuple[Database, Storage] = Depends(services)):
    db, _ = svc
    share = db.get_share(token)
    analysis = db.get_analysis(share["analysis_id"]) if share else None
    if not analysis or not analysis.get("result_json"):
        raise HTTPException(404, "공유 결과가 만료되었거나 삭제되었습니다.")
    result = analysis["result_json"]
    return {"season": result["season"], "season_ko": result["season_ko"], "undertone": result["undertone"], "palette": result["palette"], "recommendations": result["recommendations"], "products": matched_products(db, result), "disclaimer": result["disclaimer"], "reliability": result.get("reliability"), "warning_messages": result.get("warning_messages", []), "reasons": result.get("reasons", []), "content": result.get("content")}


@app.get("/api/v1/public/results/{token}/qr", include_in_schema=False)
def public_result_qr(token: str, settings: Settings = Depends(get_settings), svc: tuple[Database, Storage] = Depends(services)):
    if not svc[0].get_share(token):
        raise HTTPException(404, "공유 결과가 만료되었거나 삭제되었습니다.")
    url = f"{settings.public_base_url.rstrip('/')}/api/v1/public/results/{token}"
    return StreamingResponse(BytesIO(make_qr_png(url)), media_type="image/png", headers={"Cache-Control": "no-store"})


@app.delete("/api/v1/public/results/{token}", status_code=204)
def revoke_share(token: str, svc: tuple[Database, Storage] = Depends(services)):
    if not svc[0].delete_share(token):
        raise HTTPException(404, "공유 결과를 찾을 수 없습니다.")


@app.post("/api/v1/admin/cleanup")
def run_cleanup(
    _: None = Depends(require_admin),
    svc: tuple[Database, Storage] = Depends(services),
):
    return {"deleted": cleanup_expired(*svc)}


@app.get("/api/v1/admin/stats", dependencies=[Depends(require_admin)])
def admin_stats(request: Request, svc: tuple[Database, Storage] = Depends(services)):
    return {**svc[0].stats(), "try_on_queue": request.app.state.try_on_gate.snapshot()}

