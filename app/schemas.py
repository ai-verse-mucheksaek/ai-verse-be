from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ColorSample(BaseModel):
    hex: str
    rgb: tuple[int, int, int]


class Recommendation(BaseModel):
    category: Literal["lip", "blush", "eye"]
    name: str
    color: str
    description: str


class AnalysisResult(BaseModel):
    season: Literal["spring", "summer", "autumn", "winter"]
    season_ko: str
    undertone: Literal["warm", "cool"]
    confidence: float = Field(ge=0, le=1)
    measured_skin_color: ColorSample
    metrics: dict[str, float]
    palette: list[str]
    recommendations: list[Recommendation]
    disclaimer: str


class AnalysisResponse(BaseModel):
    id: str
    status: str
    created_at: datetime
    expires_at: datetime
    result: AnalysisResult | None = None
    original_image_url: str | None = None
    error: str | None = None


class TryOnRequest(BaseModel):
    analysis_id: str
    lip_color: str | None = None
    blush_color: str | None = None
    eye_color: str | None = None
    intensity: int = Field(default=55, ge=0, le=100)

    @field_validator("lip_color", "blush_color", "eye_color")
    @classmethod
    def valid_hex(cls, value: str | None) -> str | None:
        if value is not None and (len(value) != 7 or value[0] != "#" or any(c not in "0123456789abcdefABCDEF" for c in value[1:])):
            raise ValueError("색상은 #RRGGBB 형식이어야 합니다")
        return value.upper() if value else value


class TryOnResponse(BaseModel):
    id: str
    analysis_id: str
    status: str
    provider: str
    created_at: datetime
    result_image_url: str | None = None
    before_image_url: str | None = None
    error: str | None = None


class Product(BaseModel):
    id: str
    category: Literal["lip", "blush", "eye"]
    name: str
    brand: str
    color_hex: str
    seasons: list[Literal["spring", "summer", "autumn", "winter"]]
    tester_label: str | None = None
    active: bool = True
    match_score: int | None = Field(default=None, ge=0, le=100)

    @field_validator("color_hex")
    @classmethod
    def product_color(cls, value: str) -> str:
        if len(value) != 7 or value[0] != "#" or any(c not in "0123456789abcdefABCDEF" for c in value[1:]):
            raise ValueError("색상은 #RRGGBB 형식이어야 합니다")
        return value.upper()


class ShareLinkResponse(BaseModel):
    token: str
    expires_at: datetime
    result_url: str
    qr_image_url: str


class PublicResult(BaseModel):
    season: str
    season_ko: str
    undertone: str
    palette: list[str]
    recommendations: list[Recommendation]
    products: list[Product]
    disclaimer: str

