import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterator

from app.config import Settings


SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
  id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  expires_at TEXT NOT NULL,
  status TEXT NOT NULL,
  original_path TEXT NOT NULL,
  result_json TEXT,
  error TEXT
);
CREATE TABLE IF NOT EXISTS try_ons (
  id TEXT PRIMARY KEY,
  analysis_id TEXT NOT NULL,
  created_at TEXT NOT NULL,
  status TEXT NOT NULL,
  provider TEXT NOT NULL,
  provider_task_id TEXT,
  result_path TEXT,
  request_json TEXT NOT NULL,
  error TEXT,
  FOREIGN KEY(analysis_id) REFERENCES analyses(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS products (
  id TEXT PRIMARY KEY,
  category TEXT NOT NULL,
  name TEXT NOT NULL,
  brand TEXT NOT NULL,
  color_hex TEXT NOT NULL,
  seasons_json TEXT NOT NULL,
  tester_label TEXT,
  active INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS share_links (
  token TEXT PRIMARY KEY,
  analysis_id TEXT NOT NULL,
  created_at TEXT NOT NULL,
  expires_at TEXT NOT NULL,
  FOREIGN KEY(analysis_id) REFERENCES analyses(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_analyses_expires_at ON analyses(expires_at);
CREATE INDEX IF NOT EXISTS idx_share_links_expires_at ON share_links(expires_at);
"""

SEED_PRODUCTS = [
    ("lip-spring-01", "lip", "코랄 피치 립", "부스 테스터", "#FF6F61", '["spring"]', "LIP-01"),
    ("blush-spring-01", "blush", "살구 코랄 블러셔", "부스 테스터", "#F6A57A", '["spring"]', "BLUSH-01"),
    ("eye-spring-01", "eye", "골든 베이지 아이", "부스 테스터", "#C99152", '["spring","autumn"]', "EYE-01"),
    ("lip-summer-01", "lip", "로즈 핑크 립", "부스 테스터", "#C85A7C", '["summer"]', "LIP-02"),
    ("blush-summer-01", "blush", "라벤더 핑크 블러셔", "부스 테스터", "#D9A0BB", '["summer"]', "BLUSH-02"),
    ("eye-summer-01", "eye", "소프트 모브 아이", "부스 테스터", "#9A7F9D", '["summer"]', "EYE-02"),
    ("lip-autumn-01", "lip", "브릭 레드 립", "부스 테스터", "#A63D40", '["autumn"]', "LIP-03"),
    ("blush-autumn-01", "blush", "테라코타 블러셔", "부스 테스터", "#C66B4E", '["autumn"]', "BLUSH-03"),
    ("eye-autumn-01", "eye", "웜 브라운 아이", "부스 테스터", "#795548", '["autumn"]', "EYE-03"),
    ("lip-winter-01", "lip", "체리 레드 립", "부스 테스터", "#B51246", '["winter"]', "LIP-04"),
    ("blush-winter-01", "blush", "쿨 베리 블러셔", "부스 테스터", "#B85C7A", '["winter"]', "BLUSH-04"),
    ("eye-winter-01", "eye", "딥 플럼 아이", "부스 테스터", "#5E315B", '["winter"]', "EYE-04"),
    ("romand-bare-apricot", "lip", "베어 애프리콧", "rom&nd", "#D98C73", '["spring"]', "봄 웜톤 추천"),
    ("romand-figfig", "lip", "피그피그", "rom&nd", "#A65B70", '["summer"]', "여름 쿨톤 추천"),
    ("romand-acorn-brown", "lip", "도토리 밤", "rom&nd", "#8F4E3B", '["autumn"]', "가을 웜톤 추천"),
    ("romand-plum-coke", "lip", "플럼 콕", "rom&nd", "#802D50", '["winter"]', "겨울 쿨톤 추천"),
]


class Database:
    def __init__(self, settings: Settings):
        self.path = Path(settings.database_path)
        self.ttl_hours = settings.photo_ttl_hours

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(self.path)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        try:
            yield con
            con.commit()
        finally:
            con.close()

    def init(self) -> None:
        with self.connect() as con:
            con.executescript(SCHEMA)
            con.executemany(
                "INSERT OR IGNORE INTO products(id,category,name,brand,color_hex,seasons_json,tester_label) VALUES(?,?,?,?,?,?,?)",
                SEED_PRODUCTS,
            )

    def create_analysis(self, analysis_id: str, original_path: str) -> dict[str, Any]:
        now = datetime.now(UTC)
        row = {
            "id": analysis_id,
            "created_at": now.isoformat(),
            "expires_at": (now + timedelta(hours=self.ttl_hours)).isoformat(),
            "status": "processing",
            "original_path": original_path,
        }
        with self.connect() as con:
            con.execute(
                "INSERT INTO analyses(id, created_at, expires_at, status, original_path) VALUES(:id,:created_at,:expires_at,:status,:original_path)",
                row,
            )
        return row

    def finish_analysis(self, analysis_id: str, result: dict[str, Any]) -> None:
        with self.connect() as con:
            con.execute("UPDATE analyses SET status='completed', result_json=? WHERE id=?", (json.dumps(result, ensure_ascii=False), analysis_id))

    def fail_analysis(self, analysis_id: str, error: str) -> None:
        with self.connect() as con:
            con.execute("UPDATE analyses SET status='failed', error=? WHERE id=?", (error, analysis_id))

    def get_analysis(self, analysis_id: str) -> dict[str, Any] | None:
        with self.connect() as con:
            row = con.execute("SELECT * FROM analyses WHERE id=?", (analysis_id,)).fetchone()
        return self._decode(row, "result_json")

    def create_try_on(self, row: dict[str, Any]) -> None:
        with self.connect() as con:
            con.execute(
                "INSERT INTO try_ons(id,analysis_id,created_at,status,provider,request_json) VALUES(?,?,?,?,?,?)",
                (row["id"], row["analysis_id"], row["created_at"], row["status"], row["provider"], json.dumps(row["request"], ensure_ascii=False)),
            )

    def finish_try_on(self, try_on_id: str, result_path: str, provider_task_id: str | None = None) -> None:
        with self.connect() as con:
            con.execute("UPDATE try_ons SET status='completed', result_path=?, provider_task_id=? WHERE id=?", (result_path, provider_task_id, try_on_id))

    def fail_try_on(self, try_on_id: str, error: str, provider_task_id: str | None = None) -> None:
        with self.connect() as con:
            con.execute("UPDATE try_ons SET status='failed', error=?, provider_task_id=? WHERE id=?", (error, provider_task_id, try_on_id))

    def get_try_on(self, try_on_id: str) -> dict[str, Any] | None:
        with self.connect() as con:
            row = con.execute("SELECT * FROM try_ons WHERE id=?", (try_on_id,)).fetchone()
        return self._decode(row, "request_json")

    def delete_analysis(self, analysis_id: str) -> list[str]:
        with self.connect() as con:
            row = con.execute("SELECT original_path FROM analyses WHERE id=?", (analysis_id,)).fetchone()
            paths = [row[0]] if row else []
            paths += [r[0] for r in con.execute("SELECT result_path FROM try_ons WHERE analysis_id=? AND result_path IS NOT NULL", (analysis_id,))]
            con.execute("DELETE FROM analyses WHERE id=?", (analysis_id,))
        return paths

    def expired_ids(self) -> list[str]:
        with self.connect() as con:
            return [r[0] for r in con.execute("SELECT id FROM analyses WHERE expires_at < ?", (datetime.now(UTC).isoformat(),))]

    def list_products(self, season: str | None = None, category: str | None = None) -> list[dict[str, Any]]:
        query, params = "SELECT * FROM products WHERE active=1", []
        if category:
            query += " AND category=?"
            params.append(category)
        rows = []
        with self.connect() as con:
            for row in con.execute(query, params):
                item = dict(row)
                item["seasons"] = json.loads(item.pop("seasons_json"))
                item["active"] = bool(item["active"])
                if season is None or season in item["seasons"]:
                    rows.append(item)
        return rows

    def upsert_product(self, product: dict[str, Any]) -> None:
        with self.connect() as con:
            con.execute(
                """INSERT INTO products(id,category,name,brand,color_hex,seasons_json,tester_label,active)
                VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET category=excluded.category,name=excluded.name,
                brand=excluded.brand,color_hex=excluded.color_hex,seasons_json=excluded.seasons_json,
                tester_label=excluded.tester_label,active=excluded.active""",
                (product["id"], product["category"], product["name"], product["brand"], product["color_hex"], json.dumps(product["seasons"]), product.get("tester_label"), int(product.get("active", True))),
            )

    def create_share(self, token: str, analysis_id: str, ttl_minutes: int) -> dict[str, Any]:
        now = datetime.now(UTC)
        row = {"token": token, "analysis_id": analysis_id, "created_at": now.isoformat(), "expires_at": (now + timedelta(minutes=ttl_minutes)).isoformat()}
        with self.connect() as con:
            con.execute("INSERT INTO share_links(token,analysis_id,created_at,expires_at) VALUES(:token,:analysis_id,:created_at,:expires_at)", row)
        return row

    def get_share(self, token: str) -> dict[str, Any] | None:
        with self.connect() as con:
            row = con.execute("SELECT * FROM share_links WHERE token=? AND expires_at>=?", (token, datetime.now(UTC).isoformat())).fetchone()
        return dict(row) if row else None

    def delete_share(self, token: str) -> bool:
        with self.connect() as con:
            cursor = con.execute("DELETE FROM share_links WHERE token=?", (token,))
        return cursor.rowcount > 0

    def cleanup_shares(self) -> int:
        with self.connect() as con:
            cursor = con.execute("DELETE FROM share_links WHERE expires_at < ?", (datetime.now(UTC).isoformat(),))
        return cursor.rowcount

    def stats(self) -> dict[str, Any]:
        today = datetime.now(UTC).date().isoformat()
        with self.connect() as con:
            analyses = dict(con.execute("SELECT COUNT(*) total, SUM(status='completed') completed, SUM(status='failed') failed FROM analyses").fetchone())
            try_ons = dict(con.execute("SELECT COUNT(*) total, SUM(status='completed') completed, SUM(status='failed') failed FROM try_ons").fetchone())
            today_count = con.execute("SELECT COUNT(*) FROM analyses WHERE created_at LIKE ?", (f"{today}%",)).fetchone()[0]
            seasons = {r[0]: r[1] for r in con.execute("SELECT json_extract(result_json,'$.season'), COUNT(*) FROM analyses WHERE result_json IS NOT NULL GROUP BY 1")}
            shares = con.execute("SELECT COUNT(*) FROM share_links WHERE expires_at>=?", (datetime.now(UTC).isoformat(),)).fetchone()[0]
        return {"analyses": analyses, "try_ons": try_ons, "today_analyses": today_count, "season_counts": seasons, "active_share_links": shares}

    @staticmethod
    def _decode(row: sqlite3.Row | None, json_field: str) -> dict[str, Any] | None:
        if row is None:
            return None
        data = dict(row)
        if data.get(json_field):
            data[json_field] = json.loads(data[json_field])
        return data

