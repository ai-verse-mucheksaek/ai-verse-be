# 예뻐질 사람? It's Me! — Backend

부스용 얼굴 사진 업로드, 퍼스널컬러 분석, 메이크업 색상 추천, 가상 메이크업 Before/After를 제공하는 FastAPI 백엔드입니다.

## 포함 기능

- JPG/PNG, 파일 크기, 최소 해상도, 정면 얼굴 검증
- 얼굴의 이마·양 볼 영역을 이용한 색상 분석과 사계절 타입 분류
- 립·블러셔·아이 컬러 팔레트 추천
- 로컬 데모용 가상 메이크업 렌더러
- Perfect Corp. AI Makeup Virtual Try-On v2 연동 어댑터
- 만료형 결과 공유 링크 및 QR 코드
- 퍼스널컬러·색상 거리 기반 부스 제품 매칭
- 관리자 통계, 동시 처리 제어, 요청 제한 및 개인정보 비포함 접근 로그
- SQLite 결과 저장, 추측 불가능한 ID, `Cache-Control: no-store`
- 동의 필수, 기본 24시간 보관, 즉시 삭제 및 만료 정리
- OpenAPI 문서와 테스트

> 이 분석은 조명·카메라·촬영 각도에 영향을 받는 체험용 휴리스틱입니다. 전문 퍼스널컬러 진단이나 피부/의료 진단이 아닙니다.

## 빠른 실행

Python 3.11 이상과 [uv](https://docs.astral.sh/uv/)가 필요합니다.

```powershell
Copy-Item .env.example .env
uv sync
uv run uvicorn app.main:app --reload
```

- API 문서: <http://localhost:8000/docs>
- 상태 확인: <http://localhost:8000/health>

Docker를 쓰는 경우:

```powershell
Copy-Item .env.example .env
docker compose up --build
```

## 체험 흐름

### 1. 분석

```bash
curl -X POST http://localhost:8000/api/v1/analyses \
  -F "consent=true" \
  -F "image=@selfie.jpg"
```

응답의 `result`에는 계절 타입, 신뢰도, 측정색, 팔레트 및 메이크업 추천이 들어갑니다.

### 2. 추천 색상으로 가상 메이크업

```bash
curl -X POST http://localhost:8000/api/v1/try-ons \
  -H "Content-Type: application/json" \
  -d '{"analysis_id":"분석_ID","intensity":55}'
```

직접 색상을 고르려면 `lip_color`, `blush_color`, `eye_color`에 `#RRGGBB` 값을 추가합니다. 응답의 `before_image_url`과 `result_image_url`을 프런트에서 나란히 표시하면 됩니다.

### 3. 즉시 삭제

```bash
curl -X DELETE http://localhost:8000/api/v1/analyses/분석_ID
```

원본과 해당 분석에서 생성된 모든 결과 이미지가 함께 삭제됩니다.

### 4. 제품 추천과 QR 결과

- `GET /api/v1/analyses/{analysis_id}/products`: 분석 타입과 추천색에 가까운 부스 제품 조회
- `POST /api/v1/analyses/{analysis_id}/share`: 기본 30분 동안 유효한 결과 링크와 QR URL 발급
- `GET /api/v1/public/results/{token}`: 얼굴 사진을 제외한 공유 결과 조회
- `GET /api/v1/public/results/{token}/qr`: QR PNG 조회

### 5. 운영자 기능

`X-Admin-Token` 헤더가 필요합니다.

- `GET /api/v1/admin/stats`: 분석·가상 메이크업·계절 유형·처리 대기 현황
- `PUT /api/v1/admin/products/{product_id}`: 부스 제품 등록 및 수정
- `POST /api/v1/admin/cleanup`: 만료 데이터 즉시 정리

## 실제 YouCam API 연결

`.env`를 다음처럼 변경합니다.

```dotenv
VTO_PROVIDER=perfectcorp
PERFECTCORP_API_KEY=발급받은_API_KEY
```

서버는 공식 v2 흐름에 따라 파일 업로드 URL 발급 → 이미지 PUT → `/s2s/v2.0/task/makeup-vto` 작업 생성 → 상태 폴링 → 결과 저장을 수행합니다. 공식 문서상 입력은 JPG/JPEG/PNG, 10MB 미만, 긴 변 1920px 미만이며 얼굴 폭은 100px 이상이어야 합니다. 구현 참고: [AI Makeup Virtual Try-On 개요](https://docs.perfectcorp.com/reference/makeup_vto/section/overview), [v1.0 엔드포인트 명세](https://docs.perfectcorp.com/reference/makeup_vto/v1.0).

공급사의 패턴 카탈로그나 응답 스키마가 계정/버전에 따라 달라질 수 있으므로, 행사 전에 실제 키로 한 번 통합 테스트하고 필요하면 `app/services/virtual_tryon.py`의 패턴명을 콘솔에서 제공하는 값으로 조정하세요.

## 운영 체크리스트

- 촬영 동의 문구와 삭제 시점을 화면에 명확히 표시
- HTTPS 리버스 프록시 뒤에서 실행
- `/api/v1/admin/cleanup`은 `X-Admin-Token` 헤더가 필요합니다. 관리자 네트워크의 스케줄러에서만 호출
- 행사 종료 후 Docker 볼륨 또는 `data/`의 잔존 파일 삭제 확인
- 실제 API 사용 시 공급사 개인정보 처리방침과 보관 기간을 동의문에 반영
- 노트북 카메라 촬영 화면에서는 정면, 균일한 조명, 필터/보정 해제를 안내

## 테스트

```powershell
uv run pytest -q
```

