#!/usr/bin/env python3
"""Phase 5 smoke test: GUI settings and inference settings are separated."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.app.settings_loader import load_gui_local_settings, load_inference_runtime_settings
from src.models import AudioModel


def _check(label: str, condition: bool, detail: str = "") -> None:
    status = "[ok]" if condition else "[FAIL]"
    print(f"{status} {label}{' - ' + detail if detail else ''}")
    if not condition:
        raise SystemExit(f"failed: {label}")


def main() -> None:
    print("=== Phase 5 smoke test: settings separation ===\n")

    gui_settings = load_gui_local_settings()
    inf_settings = load_inference_runtime_settings()

    _check("gui settings loaded", gui_settings.window_width > 0)
    _check("inference settings loaded", inf_settings.backend != "")

    model = AudioModel(gui_settings=gui_settings, inference_settings=inf_settings)

    _check("samplerate from GUI settings", model.samplerate == gui_settings.samplerate)
    _check("blocksize from GUI settings", model.blocksize == gui_settings.blocksize)
    _check("local pitch from GUI settings", model.pitch_shift == gui_settings.initial_pitch_shift)
    _check(
        "rvc timeout from GUI settings",
        abs(model.rvc_processing_timeout - gui_settings.rvc_processing_timeout_sec) < 1e-6,
    )

    model.set_rvc_model("example-model")
    model.set_rvc_pitch_shift(7)
    current = model.get_current_inference_settings()

    _check("inference model name separated", current.model_name == "example-model")
    _check("inference pitch separated", current.pitch_shift == 7)
    _check("inference f0_method kept", current.f0_method == inf_settings.f0_method)
    _check("inference backend kept", current.backend == inf_settings.backend)

    print("\n=== Phase 5 smoke test passed ===")


if __name__ == "__main__":
    main()
