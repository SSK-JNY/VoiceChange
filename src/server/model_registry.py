"""RVC model discovery helpers for the WSL inference server."""

from pathlib import Path
from typing import Dict, List, Optional

from src.protocol import ModelInfo


class ModelRegistry:
    """Discovers RVC checkpoints stored under src/models/rvc."""

    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = models_dir or Path(__file__).resolve().parent.parent / "models" / "rvc"

    def list_models(self) -> List[ModelInfo]:
        models: List[ModelInfo] = []
        if not self.models_dir.exists():
            return models

        for model_path in sorted(self.models_dir.glob("*.pth")):
            stem = model_path.stem
            index_path = model_path.with_suffix(".index")
            models.append(
                ModelInfo(
                    name=stem,
                    has_index=index_path.exists(),
                    version="v2",
                )
            )
        return models

    def get_model_paths(self, model_name: str) -> Optional[Dict[str, Optional[str]]]:
        model_path = self.models_dir / f"{model_name}.pth"
        if not model_path.exists():
            return None

        index_path = model_path.with_suffix(".index")
        return {
            "name": model_name,
            "pth": str(model_path),
            "index": str(index_path) if index_path.exists() else None,
            "version": "v2",
        }