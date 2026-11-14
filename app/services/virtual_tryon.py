import asyncio
import time
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from app.config import Settings


def _hex_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i:i+2], 16) for i in (0, 2, 4))


class MockTryOnProvider:
    name = "mock"

    async def render(self, source: Path, output: Path, colors: dict[str, str], intensity: int, face_box: dict[str, int]) -> str | None:
        image = Image.open(source).convert("RGBA")
        layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        x, y, w, h = (face_box[k] for k in ("x", "y", "w", "h"))
        alpha = max(0, min(180, round(intensity * 1.8)))

        # Approximate landmarks are deliberately only a demo fallback, not a biometric model.
        blush = (*_hex_rgb(colors["blush"]), int(alpha * .35))
        r = int(w * .12)
        for cx in (x + int(w * .27), x + int(w * .73)):
            cy = y + int(h * .62)
            draw.ellipse((cx-r, cy-r//2, cx+r, cy+r//2), fill=blush)

        eye = (*_hex_rgb(colors["eye"]), int(alpha * .55))
        for cx in (x + int(w * .32), x + int(w * .68)):
            cy = y + int(h * .43)
            draw.ellipse((cx-int(w*.12), cy-int(h*.035), cx+int(w*.12), cy+int(h*.025)), fill=eye)

        lip = (*_hex_rgb(colors["lip"]), alpha)
        cx, cy = x + w // 2, y + int(h * .78)
        draw.ellipse((cx-int(w*.15), cy-int(h*.035), cx+int(w*.15), cy+int(h*.045)), fill=lip)
        layer = layer.filter(ImageFilter.GaussianBlur(max(1, int(w * .012))))
        Image.alpha_composite(image, layer).convert("RGB").save(output, "JPEG", quality=94)
        return None


class PerfectCorpProvider:
    name = "perfectcorp"

    def __init__(self, settings: Settings):
        if not settings.perfectcorp_api_key:
            raise RuntimeError("PERFECTCORP_API_KEY가 설정되지 않았습니다.")
        self.base = settings.perfectcorp_base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {settings.perfectcorp_api_key}"}
        self.interval = settings.perfectcorp_poll_interval_seconds
        self.timeout = settings.perfectcorp_poll_timeout_seconds

    async def render(self, source: Path, output: Path, colors: dict[str, str], intensity: int, face_box: dict[str, int]) -> str | None:
        content_type = "image/png" if source.suffix.lower() == ".png" else "image/jpeg"
        async with httpx.AsyncClient(timeout=30) as client:
            init = await client.post(f"{self.base}/s2s/v2.0/file", headers=self.headers, json={"files": [{"content_type": content_type, "file_name": source.name, "file_size": source.stat().st_size}]})
            init.raise_for_status()
            file_info = init.json()["data"]["files"][0]
            request = file_info["requests"][0]
            upload_headers = dict(request.get("headers", {}))
            upload_headers["Content-Type"] = content_type
            uploaded = await client.request(request.get("method", "PUT"), request["url"], headers=upload_headers, content=source.read_bytes())
            uploaded.raise_for_status()

            palette = lambda color: {"color": color, "texture": "matte", "colorIntensity": intensity}
            effects = [
                {"category": "lip_color", "shape": {"name": "original"}, "style": {"type": "full"}, "palettes": [palette(colors["lip"])]},
                {"category": "blush", "pattern": {"name": "1color1"}, "palettes": [palette(colors["blush"])]},
                {"category": "eye_shadow", "pattern": {"name": "1color1"}, "palettes": [palette(colors["eye"])]},
            ]
            task = await client.post(f"{self.base}/s2s/v2.0/task/makeup-vto", headers={**self.headers, "Content-Type": "application/json"}, json={"src_file_id": file_info["file_id"], "version": "1.0", "effects": effects})
            task.raise_for_status()
            task_id = task.json()["data"]["task_id"]
            deadline = time.monotonic() + self.timeout
            while time.monotonic() < deadline:
                status = await client.get(f"{self.base}/s2s/v2.0/task/makeup-vto/{task_id}", headers=self.headers)
                status.raise_for_status()
                data = status.json().get("data", {})
                state = data.get("task_status") or data.get("status")
                if state in {"success", "completed"}:
                    result_url = (data.get("results") or {}).get("url") or data.get("result_url")
                    if not result_url:
                        raise RuntimeError("공급사 응답에 결과 이미지 URL이 없습니다.")
                    result = await client.get(result_url)
                    result.raise_for_status()
                    output.write_bytes(result.content)
                    return task_id
                if state in {"error", "failed"}:
                    raise RuntimeError(data.get("error") or "가상 메이크업 처리에 실패했습니다.")
                await asyncio.sleep(self.interval)
            raise TimeoutError("가상 메이크업 처리 시간이 초과되었습니다.")


def provider_for(settings: Settings):
    return PerfectCorpProvider(settings) if settings.vto_provider == "perfectcorp" else MockTryOnProvider()

