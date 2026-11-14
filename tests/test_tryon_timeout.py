
import pytest
import httpx

from app.config import Settings
from app.services.virtual_tryon import PerfectCorpProvider


@pytest.mark.asyncio
async def test_youcam_timeout(monkeypatch, tmp_path):
    # YouCam 서버를 대신하는 가짜 응답
    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, url, **kwargs):
            if url.endswith("/file"):
                data = {
                    "data": {
                        "files": [{
                            "file_id": "fake-file",
                            "requests": [{
                                "url": "https://example.com/upload",
                                "method": "PUT"
                            }]
                        }]
                    }
                }
            else:
                data = {"data": {"task_id": "fake-task"}}

            return httpx.Response(
                200,
                json=data,
                request=httpx.Request("POST", url)
            )

        async def request(self, method, url, **kwargs):
            return httpx.Response(
                200,
                request=httpx.Request(method, url)
            )

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: FakeClient())

    settings = Settings(
        perfectcorp_api_key="fake-key",
        perfectcorp_poll_timeout_seconds=0
    )
    provider = PerfectCorpProvider(settings)

    source = tmp_path / "face.jpg"
    source.write_bytes(b"fake-image")
    output = tmp_path / "result.jpg"

    colors = {
        "lip": "#FF6699",
        "blush": "#FFAAAA",
        "eye": "#AA8866"
    }

    with pytest.raises(TimeoutError):
        await provider.render(
            source, output, colors, 55,
            {"x": 0, "y": 0, "w": 100, "h": 100}
        )
