"""設定ファイルのロード処理。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .gui_local_settings import GuiLocalSettings
from .inference_runtime_settings import InferenceRuntimeSettings


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_GUI_SETTINGS_PATH = _PROJECT_ROOT / "gui_local_settings.json"
_INFERENCE_SETTINGS_PATH = _PROJECT_ROOT / "inference_settings.json"


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_gui_local_settings(path: Optional[Path] = None) -> GuiLocalSettings:
    target = path or _GUI_SETTINGS_PATH
    if not target.exists():
        defaults = GuiLocalSettings()
        _write_json(target, defaults.to_dict())
        return defaults
    return GuiLocalSettings.from_dict(_read_json(target))


def load_inference_runtime_settings(path: Optional[Path] = None) -> InferenceRuntimeSettings:
    target = path or _INFERENCE_SETTINGS_PATH
    if not target.exists():
        defaults = InferenceRuntimeSettings()
        _write_json(target, defaults.to_dict())
        return defaults
    return InferenceRuntimeSettings.from_dict(_read_json(target))
