import os
from pathlib import Path

from .config import SEED, Paths
from .dataset import Manifest
from .utils import ProjectError, read_json, sha256_file, write_json

ARTIFACT_VERSION = 1


def publish_model(staged: Path, paths: Paths, model_key: str, manifest: Manifest,
                  preprocess: dict, train_seconds: float, details: dict) -> dict:
    metadata = {
        "version": ARTIFACT_VERSION,
        "model_key": model_key,
        "model_sha256": sha256_file(staged),
        "dataset_id": manifest.dataset_id,
        "class_names": manifest.class_names,
        "class_to_index": manifest.class_to_index,
        "preprocess": preprocess,
        "seed": SEED,
        "train_seconds": train_seconds,
        "training": details,
    }
    os.replace(staged, paths.model_file(model_key))
    write_json(paths.metadata_file(model_key), metadata)
    return metadata


def read_model_metadata(paths: Paths, model_key: str) -> dict:
    if not paths.model_file(model_key).is_file() or not paths.metadata_file(model_key).is_file():
        raise ProjectError(f"Chưa có mô hình {model_key}. Hãy huấn luyện để tạo mô hình và tệp JSON đi kèm.")
    metadata = read_json(paths.metadata_file(model_key))
    names = metadata.get("class_names")
    if (
        metadata.get("version") != ARTIFACT_VERSION
        or metadata.get("model_key") != model_key
        or not isinstance(names, list)
        or len(names) < 2
        or metadata.get("class_to_index") != {name: index for index, name in enumerate(names)}
    ):
        raise ProjectError(f"Metadata của {model_key} không hợp lệ.")
    return metadata


def model_signature(paths: Paths, model_key: str) -> tuple[int, int]:
    return (paths.model_file(model_key).stat().st_mtime_ns, paths.metadata_file(model_key).stat().st_mtime_ns)
