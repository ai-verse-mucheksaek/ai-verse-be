import os
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from app.config import Settings


ALLOWED_TYPES = {"image/jpeg": ".jpg", "image/png": ".png"}


class Storage:
    def __init__(self, settings: Settings):
        self.root = Path(settings.storage_dir)
        self.uploads = self.root / "uploads"
        self.results = self.root / "results"
        self.max_bytes = settings.max_upload_mb * 1024 * 1024
        self.uploads.mkdir(parents=True, exist_ok=True)
        self.results.mkdir(parents=True, exist_ok=True)

    async def save_upload(self, upload: UploadFile) -> Path:
        if upload.content_type not in ALLOWED_TYPES:
            raise ValueError("JPG 또는 PNG 이미지만 업로드할 수 있습니다.")
        path = self.uploads / f"{uuid4().hex}{ALLOWED_TYPES[upload.content_type]}"
        size = 0
        try:
            with path.open("wb") as target:
                while chunk := await upload.read(1024 * 1024):
                    size += len(chunk)
                    if size > self.max_bytes:
                        raise ValueError(f"이미지는 {self.max_bytes // 1024 // 1024}MB 이하여야 합니다.")
                    target.write(chunk)
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return path

    @staticmethod
    def delete(paths: list[str]) -> None:
        for raw in paths:
            try:
                Path(raw).unlink(missing_ok=True)
            except OSError:
                pass

    @staticmethod
    def safe_file(path: str | None, expected_parent: Path) -> Path | None:
        if not path:
            return None
        resolved = Path(path).resolve()
        if resolved.parent != expected_parent.resolve() or not resolved.is_file():
            return None
        return resolved

