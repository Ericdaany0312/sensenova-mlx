from __future__ import annotations

import base64
import json
import threading
import unittest
from http.client import HTTPConnection

from sensenova_mlx.server import (
    GeneratedImage,
    ImageGenerationService,
    ImageRequest,
    create_server,
)


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl2ZQAAAABJRU5ErkJggg=="
)


class RecordingGenerator:
    def __init__(self) -> None:
        self.requests: list[ImageRequest] = []

    def __call__(self, request: ImageRequest) -> GeneratedImage:
        self.requests.append(request)
        return GeneratedImage(png=PNG_1X1, elapsed_seconds=0.25, seed=request.seed)


class BlockingGenerator:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def __call__(self, request: ImageRequest) -> GeneratedImage:
        self.started.set()
        self.release.wait(timeout=3)
        return GeneratedImage(png=PNG_1X1, elapsed_seconds=1.0, seed=request.seed)


class SenseNovaImageServerTest(unittest.TestCase):
    API_KEY = "local-test-key-123456"

    def setUp(self) -> None:
        self.generator = RecordingGenerator()
        self.httpd = create_server(
            "127.0.0.1",
            0,
            api_key=self.API_KEY,
            service=ImageGenerationService(self.generator),
        )
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=2)

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, object] | None = None,
        *,
        authorized: bool = True,
    ) -> tuple[int, dict[str, object]]:
        connection = HTTPConnection("127.0.0.1", self.httpd.server_port, timeout=3)
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"content-type": "application/json"}
        if authorized:
            headers["authorization"] = f"Bearer {self.API_KEY}"
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        parsed = json.loads(response.read().decode("utf-8"))
        connection.close()
        return response.status, parsed

    def test_health_is_public_and_does_not_expose_configuration(self) -> None:
        status, payload = self.request("GET", "/health", authorized=False)

        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["model"], "SenseNova-U1.5-8B-MoT-LoRA-8step")
        self.assertNotIn("api_key", payload)

    def test_generation_requires_bearer_authentication(self) -> None:
        status, payload = self.request(
            "POST",
            "/v1/images/generations",
            {"model": "SenseNova-U1.5-8B-MoT-LoRA-8step", "prompt": "会议海报"},
            authorized=False,
        )

        self.assertEqual(status, 401)
        self.assertEqual(payload["error"]["code"], "invalid_api_key")
        self.assertEqual(self.generator.requests, [])

    def test_openai_compatible_generation_returns_base64_png_and_metadata(self) -> None:
        status, payload = self.request(
            "POST",
            "/v1/images/generations",
            {
                "model": "SenseNova-U1.5-8B-MoT-LoRA-8step",
                "prompt": "竖版企业会议主视觉，不要文字",
                "size": "1760x2368",
                "n": 1,
                "seed": 20260822,
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(base64.b64decode(payload["data"][0]["b64_json"]), PNG_1X1)
        self.assertEqual(payload["model"], "SenseNova-U1.5-8B-MoT-LoRA-8step")
        self.assertEqual(payload["metadata"]["size"], "1760x2368")
        self.assertEqual(payload["metadata"]["seed"], 20260822)
        self.assertEqual(
            self.generator.requests,
            [ImageRequest(prompt="竖版企业会议主视觉，不要文字", width=1760, height=2368, seed=20260822)],
        )

    def test_generation_rejects_unsupported_size_before_model_execution(self) -> None:
        status, payload = self.request(
            "POST",
            "/v1/images/generations",
            {
                "model": "SenseNova-U1.5-8B-MoT-LoRA-8step",
                "prompt": "会议海报",
                "size": "123x456",
            },
        )

        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["code"], "invalid_size")
        self.assertEqual(self.generator.requests, [])

    def test_generation_rejects_multiple_images_before_model_execution(self) -> None:
        status, payload = self.request(
            "POST",
            "/v1/images/generations",
            {
                "model": "SenseNova-U1.5-8B-MoT-LoRA-8step",
                "prompt": "会议海报",
                "n": 2,
            },
        )

        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["code"], "invalid_n")
        self.assertEqual(self.generator.requests, [])

    def test_second_generation_is_rejected_while_gpu_is_busy(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=2)
        blocking = BlockingGenerator()
        self.httpd = create_server(
            "127.0.0.1",
            0,
            api_key=self.API_KEY,
            service=ImageGenerationService(blocking),
        )
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

        first_result: list[tuple[int, dict[str, object]]] = []
        first = threading.Thread(
            target=lambda: first_result.append(
                self.request(
                    "POST",
                    "/v1/images/generations",
                    {"model": "SenseNova-U1.5-8B-MoT-LoRA-8step", "prompt": "第一张"},
                )
            )
        )
        first.start()
        self.assertTrue(blocking.started.wait(timeout=2))

        status, payload = self.request(
            "POST",
            "/v1/images/generations",
            {"model": "SenseNova-U1.5-8B-MoT-LoRA-8step", "prompt": "第二张"},
        )
        blocking.release.set()
        first.join(timeout=3)

        self.assertEqual(status, 429)
        self.assertEqual(payload["error"]["code"], "model_busy")
        self.assertEqual(first_result[0][0], 200)


if __name__ == "__main__":
    unittest.main()
