from __future__ import annotations

import base64
import hmac
import io
import json
import os
import secrets
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Mapping


DEFAULT_MODEL_ID = "SenseNova-U1.5-8B-MoT-LoRA-8step"
MAX_REQUEST_BYTES = 32 * 1024
MAX_PROMPT_CHARS = 4_000
SUPPORTED_SIZES = frozenset(
    {
        "1024x1024",
        "1440x2560",
        "1536x2720",
        "1664x2496",
        "1760x2368",
        "2048x2048",
        "2368x1760",
        "2496x1664",
        "2560x1440",
        "2720x1536",
    }
)


@dataclass(frozen=True)
class ImageRequest:
    prompt: str
    width: int
    height: int
    seed: int


@dataclass(frozen=True)
class GeneratedImage:
    png: bytes
    elapsed_seconds: float
    seed: int


class APIError(ValueError):
    def __init__(self, status: int, message: str, code: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.code = code


class ModelBusyError(RuntimeError):
    pass


class ImageGenerationService:
    def __init__(self, generator: Callable[[ImageRequest], GeneratedImage]) -> None:
        self._generator = generator
        self._slot = threading.Semaphore(1)

    @property
    def busy(self) -> bool:
        acquired = self._slot.acquire(blocking=False)
        if acquired:
            self._slot.release()
            return False
        return True

    def generate(self, request: ImageRequest) -> GeneratedImage:
        if not self._slot.acquire(blocking=False):
            raise ModelBusyError("the image model is already processing another request")
        try:
            return self._generator(request)
        finally:
            self._slot.release()


def parse_image_request(payload: Mapping[str, object], *, model_id: str) -> ImageRequest:
    model = payload.get("model", model_id)
    if model != model_id:
        raise APIError(400, f"model must be {model_id}", "model_not_found")

    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise APIError(400, "prompt must be a non-empty string", "invalid_prompt")
    normalized_prompt = prompt.strip()
    if len(normalized_prompt) > MAX_PROMPT_CHARS:
        raise APIError(400, f"prompt must not exceed {MAX_PROMPT_CHARS} characters", "invalid_prompt")

    n = payload.get("n", 1)
    if isinstance(n, bool) or not isinstance(n, int) or n != 1:
        raise APIError(400, "only n=1 is supported", "invalid_n")

    size = payload.get("size", "1024x1024")
    if not isinstance(size, str) or size not in SUPPORTED_SIZES:
        raise APIError(400, "size is not supported by this service", "invalid_size")
    width_text, height_text = size.split("x", 1)

    seed = payload.get("seed")
    if seed is None:
        seed = secrets.randbelow(2**32)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise APIError(400, "seed must be an integer from 0 to 4294967295", "invalid_seed")

    return ImageRequest(
        prompt=normalized_prompt,
        width=int(width_text),
        height=int(height_text),
        seed=seed,
    )


class SenseNovaHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        *,
        api_key: str,
        service: ImageGenerationService,
        model_id: str,
    ) -> None:
        self.api_key = api_key
        self.service = service
        self.model_id = model_id
        super().__init__(server_address, SenseNovaRequestHandler)


class SenseNovaRequestHandler(BaseHTTPRequestHandler):
    server: SenseNovaHTTPServer

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler contract
        if self.path == "/health":
            self._send_json(
                200,
                {
                    "status": "busy" if self.server.service.busy else "ready",
                    "model": self.server.model_id,
                },
            )
            return
        if self.path == "/v1/models":
            if not self._authorized():
                return
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self.server.model_id,
                            "object": "model",
                            "owned_by": "local-sensenova-mlx",
                        }
                    ],
                },
            )
            return
        self._send_error(404, "path not found", "not_found")

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler contract
        if self.path != "/v1/images/generations":
            self._send_error(404, "path not found", "not_found")
            return
        if not self._authorized():
            return
        try:
            payload = self._read_json()
            request = parse_image_request(payload, model_id=self.server.model_id)
            generated = self.server.service.generate(request)
        except APIError as error:
            self._send_error(error.status, error.message, error.code)
            return
        except ModelBusyError:
            self._send_error(
                429,
                "the local image model is busy; retry after the current image finishes",
                "model_busy",
                extra_headers={"retry-after": "3"},
            )
            return
        except Exception as error:
            print(f"[error] generation failed: {type(error).__name__}: {error}", flush=True)
            self._send_error(500, "local image generation failed", "generation_failed")
            return

        self._send_json(
            200,
            {
                "created": int(time.time()),
                "model": self.server.model_id,
                "data": [{"b64_json": base64.b64encode(generated.png).decode("ascii")}],
                "metadata": {
                    "size": f"{request.width}x{request.height}",
                    "seed": generated.seed,
                    "elapsed_seconds": round(generated.elapsed_seconds, 3),
                },
            },
        )

    def _authorized(self) -> bool:
        supplied = self.headers.get("authorization", "")
        expected = f"Bearer {self.server.api_key}"
        if hmac.compare_digest(supplied, expected):
            return True
        self._send_error(401, "invalid API key", "invalid_api_key")
        return False

    def _read_json(self) -> Mapping[str, object]:
        content_type = self.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            raise APIError(415, "content-type must be application/json", "invalid_content_type")
        try:
            length = int(self.headers.get("content-length", "0"))
        except ValueError as error:
            raise APIError(400, "invalid content-length", "invalid_request") from error
        if length <= 0 or length > MAX_REQUEST_BYTES:
            raise APIError(413, "request body is empty or too large", "request_too_large")
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise APIError(400, "request body must be valid UTF-8 JSON", "invalid_json") from error
        if not isinstance(payload, dict):
            raise APIError(400, "request body must be a JSON object", "invalid_request")
        return payload

    def _send_error(
        self,
        status: int,
        message: str,
        code: str,
        *,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        self._send_json(
            status,
            {"error": {"message": message, "type": "invalid_request_error", "code": code}},
            extra_headers=extra_headers,
        )

    def _send_json(
        self,
        status: int,
        payload: Mapping[str, object],
        *,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("content-length", str(len(encoded)))
        self.send_header("cache-control", "no-store")
        for name, value in (extra_headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: object) -> None:
        # Keep request logs useful without ever including authorization headers or request bodies.
        print(f"[http] {self.address_string()} {format % args}")


def create_server(
    host: str,
    port: int,
    *,
    api_key: str,
    service: ImageGenerationService,
    model_id: str = DEFAULT_MODEL_ID,
) -> SenseNovaHTTPServer:
    if len(api_key) < 16:
        raise ValueError("API key must contain at least 16 characters")
    return SenseNovaHTTPServer(
        (host, port),
        api_key=api_key,
        service=service,
        model_id=model_id,
    )


class MlxImageGenerator:
    """Load the formal checkpoint once and reuse it for sequential image requests."""

    MAX_ACTIVE_BYTES = 90 * 1024**3
    MAX_SWAP_DELTA_BYTES = 4 * 1024**3

    def __init__(self, model_dir: Path, *, mode: str) -> None:
        import mlx.core as mx
        from transformers import AutoTokenizer

        from .generate import _load_pipeline, _swap_used_bytes

        self._mx = mx
        self._swap_used_bytes = _swap_used_bytes
        self._mode = mode
        self._steps = 8 if mode == "fast" else 50
        self._cfg_scale = 1.0 if mode == "fast" else 4.0
        self.model_id = (
            DEFAULT_MODEL_ID if mode == "fast" else "SenseNova-U1.5-8B-MoT"
        )

        swap_before = _swap_used_bytes()
        self._pipeline, self._config = _load_pipeline(model_dir, use_lora=mode == "fast")
        if mx.get_active_memory() > self.MAX_ACTIVE_BYTES:
            raise MemoryError("formal weights exceeded the MLX active-memory stop limit")
        swap_after = _swap_used_bytes()
        if (
            swap_before is not None
            and swap_after is not None
            and swap_after - swap_before > self.MAX_SWAP_DELTA_BYTES
        ):
            raise MemoryError("swap increased by more than 4 GiB while loading formal weights")
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            local_files_only=True,
            trust_remote_code=False,
        )

    def __call__(self, request: ImageRequest) -> GeneratedImage:
        import numpy as np
        from PIL import Image

        from .prompt import build_t2i_query

        started = time.monotonic()
        swap_before = self._swap_used_bytes()
        self._mx.reset_peak_memory()
        self._config.validate_image_size(request.width, request.height)

        def token_ids(prompt: str, *, conditional: bool):
            query = build_t2i_query(prompt, conditional=conditional)
            values = self._tokenizer.encode(query, add_special_tokens=True)
            if not values:
                raise ValueError("tokenizer produced an empty prefix")
            return self._mx.array([values], dtype=self._mx.int32)

        condition_ids = token_ids(request.prompt, conditional=True)
        uncondition_ids = (
            token_ids("", conditional=False) if self._cfg_scale > 1.0 else None
        )

        def report_step(completed: int, total: int) -> None:
            current_swap = self._swap_used_bytes()
            if (
                swap_before is not None
                and current_swap is not None
                and current_swap - swap_before > self.MAX_SWAP_DELTA_BYTES
            ):
                raise MemoryError("swap increased by more than 4 GiB during image generation")
            print(
                f"[step] {completed}/{total} "
                f"active={self._mx.get_active_memory() / 1024**3:.2f}GiB "
                f"peak={self._mx.get_peak_memory() / 1024**3:.2f}GiB",
                flush=True,
            )

        image = self._pipeline.generate_from_ids(
            condition_ids,
            uncondition_ids,
            width=request.width,
            height=request.height,
            num_steps=self._steps,
            seed=request.seed,
            cfg_scale=self._cfg_scale,
            cfg_norm="none",
            timestep_shift=3.0,
            max_active_bytes=self.MAX_ACTIVE_BYTES,
            step_callback=report_step,
        )
        self._mx.eval(image)
        array = np.asarray(image.astype(self._mx.float32))
        if array.shape != (1, 3, request.height, request.width):
            raise ValueError(f"generated image has unexpected shape {array.shape}")
        if not np.isfinite(array).all():
            raise FloatingPointError("generated image contains NaN or infinity")
        pixels = np.clip(array[0] * 0.5 + 0.5, 0.0, 1.0)
        uint8 = np.rint(pixels.transpose(1, 2, 0) * 255.0).astype(np.uint8)
        if int(uint8.min()) == int(uint8.max()):
            raise ValueError("generated image is a single flat color")
        output = io.BytesIO()
        Image.fromarray(uint8).save(output, format="PNG")
        del image, array, pixels, uint8
        self._mx.clear_cache()
        return GeneratedImage(
            png=output.getvalue(),
            elapsed_seconds=time.monotonic() - started,
            seed=request.seed,
        )


def serve_from_args(args: object) -> None:
    host = str(getattr(args, "host"))
    port = int(getattr(args, "port"))
    api_key_env = str(getattr(args, "api_key_env"))
    api_key = os.environ.get(api_key_env, "")
    if len(api_key) < 16:
        raise ValueError(f"{api_key_env} must contain an API key with at least 16 characters")

    generator = MlxImageGenerator(Path(getattr(args, "model_dir")), mode=str(getattr(args, "mode")))
    httpd = create_server(
        host,
        port,
        api_key=api_key,
        service=ImageGenerationService(generator),
        model_id=generator.model_id,
    )
    print(f"[ready] http://{host}:{httpd.server_port} model={generator.model_id}", flush=True)
    try:
        httpd.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
