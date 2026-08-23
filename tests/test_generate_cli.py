from __future__ import annotations

import unittest
from unittest.mock import patch
from pathlib import Path

from sensenova_mlx.generate import (
    _effective_cfg_scale,
    _mlx_version,
    build_parser,
    default_model_dir,
    validate_args,
)


class GenerateCLITest(unittest.TestCase):
    def test_default_model_dir_prefers_explicit_environment_override(self) -> None:
        with patch.dict(
            "os.environ",
            {"SENSENOVA_MODEL_DIR": "/opt/models/sensenova", "XDG_CACHE_HOME": "/tmp/cache"},
            clear=True,
        ):
            self.assertEqual(default_model_dir(), Path("/opt/models/sensenova"))

    def test_default_model_dir_uses_xdg_cache_when_available(self) -> None:
        with patch.dict("os.environ", {"XDG_CACHE_HOME": "/tmp/cache"}, clear=True):
            self.assertEqual(
                default_model_dir(),
                Path("/tmp/cache/sensenova-mlx/SenseNova-U1.5-8B-MoT"),
            )

    def test_default_model_dir_falls_back_to_user_cache(self) -> None:
        with patch.dict("os.environ", {}, clear=True), patch(
            "sensenova_mlx.generate.Path.home", return_value=Path("/" + "Users/tester")
        ):
            self.assertEqual(
                default_model_dir(),
                Path("/" + "Users/tester/.cache/sensenova-mlx/SenseNova-U1.5-8B-MoT"),
            )

    def test_generate_parser_defaults_to_formal_model(self) -> None:
        with patch.dict("os.environ", {"SENSENOVA_MODEL_DIR": "/models/formal"}, clear=True):
            args = build_parser().parse_args(["generate", "--prompt", "a red fox"])
            validate_args(args)

        self.assertEqual(args.model_dir, Path("/models/formal"))
        self.assertEqual(args.width, 1024)
        self.assertEqual(args.height, 1024)
        self.assertEqual(args.steps, 50)
        self.assertFalse(args.lora_8step)
        self.assertEqual(_effective_cfg_scale(args), 4.0)

    def test_formal_eight_step_lora_defaults_to_official_cfg_one(self) -> None:
        args = build_parser().parse_args(
            ["generate", "--prompt", "a red fox", "--lora-8step", "--steps", "8"]
        )
        validate_args(args)

        self.assertEqual(_effective_cfg_scale(args), 1.0)

    def test_eight_step_adapter_cannot_silently_run_with_wrong_schedule(self) -> None:
        args = build_parser().parse_args(
            ["generate", "--prompt", "a red fox", "--lora-8step", "--steps", "50"]
        )
        with self.assertRaisesRegex(ValueError, "exactly --steps 8"):
            validate_args(args)

    def test_serve_parser_defaults_to_loopback_fast_model(self) -> None:
        args = build_parser().parse_args(["serve"])
        validate_args(args)

        self.assertEqual(args.host, "127.0.0.1")
        self.assertEqual(args.port, 11235)
        self.assertEqual(args.mode, "fast")
        self.assertEqual(args.api_key_env, "SENSENOVA_API_KEY")
        self.assertFalse(args.allow_remote)

    def test_serve_rejects_remote_binding_without_explicit_opt_in(self) -> None:
        args = build_parser().parse_args(["serve", "--host", "0.0.0.0"])

        with self.assertRaisesRegex(ValueError, "--allow-remote"):
            validate_args(args)

    def test_runtime_package_does_not_import_torch(self) -> None:
        package = Path(__file__).parents[1] / "src" / "sensenova_mlx"
        offenders = [
            path.name
            for path in package.glob("*.py")
            if "import torch" in path.read_text(encoding="utf-8")
        ]
        self.assertEqual(offenders, [])

    def test_mlx_version_is_available_when_top_level_module_omits_dunder_version(self) -> None:
        version = _mlx_version()

        self.assertIsInstance(version, str)
        self.assertTrue(version)


if __name__ == "__main__":
    unittest.main()
