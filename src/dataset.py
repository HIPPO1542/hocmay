"""Kiểm tra bộ dữ liệu gốc và tạo danh sách ảnh sạch dùng chung."""
import hashlib
import json
import random
import struct
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np

from .config import IMAGE_EXTENSIONS, SEED, SPLITS, Paths
from .preprocessing import read_image
from .utils import ProjectError, read_json, run_cli, sha256_file, workspace_parser, write_csv, write_json

AUDIT_FIELDS = [
    "path", "split", "effective_split", "class_name", "width", "height", "pixel_hash", "file_hash",
    "status", "detail",
]
VALIDATION_FRACTION = 0.10


@dataclass(frozen=True)
class Manifest:
    dataset_id: str
    class_names: list[str]
    images: list[dict]

    @property
    def class_to_index(self) -> dict[str, int]:
        return {name: index for index, name in enumerate(self.class_names)}

    def for_split(self, paths: Paths, split: str) -> tuple[list[Path], list[int]]:
        entries = [image for image in self.images if image["split"] == split]
        image_paths = [paths.raw / entry["path"] for entry in entries]
        labels = [entry["class_index"] for entry in entries]
        return image_paths, labels


def find_split_root(raw: Path) -> Path:
    if not raw.is_dir():
        raise ProjectError(f"Chưa có thư mục dữ liệu {raw}. Xem README để tải từ Kaggle.")
    candidates = {raw, *(folder.parent for folder in raw.rglob("train") if folder.is_dir())}
    roots = {folder for folder in candidates if all((folder / split).is_dir() for split in SPLITS)}
    if len(roots) != 1:
        raise ProjectError(f"Tìm thấy {len(roots)} bộ train/validation/test trong {raw}. Cần đúng một bộ.")
    return roots.pop()


def class_directories(split_dir: Path) -> list[Path]:
    return sorted(folder for folder in split_dir.iterdir() if folder.is_dir() and not folder.name.startswith("."))


def pixel_hash(image) -> str:
    digest = hashlib.sha256(struct.pack("!II", *image.size))
    digest.update(image.tobytes())
    return digest.hexdigest()


def image_candidates(split_root: Path):
    for split in SPLITS:
        for folder in class_directories(split_root / split):
            for path in sorted(folder.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    yield path, split, folder.name


def inspect_image(candidate: tuple[Path, str, str], *, raw: Path, known_classes: set[str]) -> dict:
    path, split, class_name = candidate
    record = dict.fromkeys(AUDIT_FIELDS, "")
    record.update(path=path.relative_to(raw).as_posix(), split=split, class_name=class_name)
    try:
        image = read_image(path)
        record.update(width=image.width, height=image.height,
                      pixel_hash=pixel_hash(image), file_hash=sha256_file(path))
        record["status"] = "kept" if class_name in known_classes else "unknown_class"
    except (ProjectError, OSError) as error:
        record.update(status="invalid", detail=str(error))
    return record


def resolve_duplicates(records: list[dict]) -> list[dict]:
    by_hash = defaultdict(list)
    for record in records:
        if record["pixel_hash"]:
            by_hash[record["pixel_hash"]].append(record)

    duplicate_groups = []
    for digest, group in by_hash.items():
        if len(group) < 2:
            continue
        splits = sorted({record["split"] for record in group}, key=SPLITS.index)
        labels = sorted({record["class_name"] for record in group})
        # Giữ ảnh ở tập có mức ưu tiên cao nhất để ảnh test không lọt vào train/validation.
        ordered = sorted(group, key=lambda record: (-SPLITS.index(record["split"]), record["path"]))
        keeper = ordered[0]
        if len(labels) > 1:
            for record in ordered:
                record.update(status="label_conflict", detail="Ảnh giống hệt pixel nhưng khác nhãn")
        else:
            for record in ordered[1:]:
                record.update(status="duplicate", detail=keeper["path"])
        duplicate_groups.append(
            {
                "pixel_hash": digest,
                "cross_split": len(splits) > 1,
                "splits": splits,
                "class_names": labels,
                "kept_path": keeper["path"] if keeper["status"] == "kept" else None,
                "paths": [record["path"] for record in ordered],
            }
        )
    return duplicate_groups


def dataset_fingerprint(images: list[dict], class_names: list[str]) -> str:
    encoded = json.dumps([class_names, images], sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def ensure_validation_coverage(kept: list[dict], class_names: list[str]) -> list[dict]:
    validation_counts = Counter(record["class_name"] for record in kept if record["split"] == "validation")
    if all(validation_counts[class_name] for class_name in class_names):
        return []

    # Chỉ bổ sung khi validation thiếu lớp sau loại trùng, lấy từ train đã tách khỏi test.
    rng = random.Random(SEED)
    train_by_class = defaultdict(list)
    for record in kept:
        if record["split"] == "train":
            train_by_class[record["class_name"]].append(record)

    moved = []
    for class_name in class_names:
        available = train_by_class[class_name]
        target = max(1, round(len(available) * VALIDATION_FRACTION))
        needed = max(0, target - validation_counts[class_name])
        count = min(needed, max(0, len(available) - 1))
        for record in rng.sample(available, count):
            record["effective_split"] = "validation"
            moved.append({"path": record["path"], "class_name": class_name})
    return moved


def prepare_dataset(paths: Paths) -> dict:
    from .eda import save_eda_figures

    split_root = find_split_root(paths.raw)
    class_names = [folder.name for folder in class_directories(split_root / "train")]
    class_index = {name: index for index, name in enumerate(class_names)}
    unknown = sorted(
        {folder.name for split in SPLITS[1:] for folder in class_directories(split_root / split)} - set(class_names)
    )
    inspect = partial(inspect_image, raw=paths.raw, known_classes=set(class_names))
    with ThreadPoolExecutor(max_workers=4) as workers:
        records = list(workers.map(inspect, image_candidates(split_root)))
    duplicate_groups = resolve_duplicates(records)
    kept = [record for record in records if record["status"] == "kept"]
    for record in kept:
        record["effective_split"] = record["split"]
    moved = ensure_validation_coverage(kept, class_names)
    totals = Counter(record["split"] for record in records)
    retained_original = Counter(record["split"] for record in kept)
    original_counts = Counter((record["split"], record["class_name"]) for record in records)
    retained = Counter((record["effective_split"], record["class_name"]) for record in kept)
    images = [
        {
            "path": record["path"],
            "split": record["effective_split"],
            "source_split": record["split"],
            "class_index": class_index[record["class_name"]],
            "pixel_hash": record["pixel_hash"],
            "file_hash": record["file_hash"],
        }
        for record in kept
    ]
    class_counts = [
        {"class_name": name, **{split: retained[split, name] for split in SPLITS},
         **{f"{split}_candidates": original_counts[split, name] for split in SPLITS}}
        for name in class_names
    ]
    train_counts = [row["train"] for row in class_counts]
    missing_train = [row["class_name"] for row in class_counts if row["train"] == 0]
    missing_validation = [row["class_name"] for row in class_counts if row["validation"] == 0]
    empty_splits = [split for split in SPLITS if not any(row[split] for row in class_counts)]
    issues = []
    if len(class_names) < 2:
        issues.append("Cần ít nhất hai lớp trong train.")
    if unknown:
        issues.append(f"Nhãn ngoài danh sách train: {', '.join(unknown)}.")
    if missing_train:
        issues.append(f"Lớp train hết ảnh sau kiểm tra: {', '.join(missing_train)}.")
    if missing_validation:
        issues.append(f"Thiếu ảnh sạch để có validation cho các lớp: {', '.join(missing_validation)}.")
    if empty_splits:
        issues.append(f"Tập không còn ảnh hợp lệ: {', '.join(empty_splits)}.")

    summary = {
        "dataset_id": dataset_fingerprint(images, class_names),
        "total_candidates": len(records),
        "total_kept": len(kept),
        "class_count": len(class_names),
        "class_names": class_names,
        "candidates_by_split": {split: totals[split] for split in SPLITS},
        "kept_by_original_split": {split: retained_original[split] for split in SPLITS},
        "kept_by_split": {split: sum(row[split] for row in class_counts) for split in SPLITS},
        "validation_repair": {
            "moved_from_train": len(moved),
            "target_fraction_per_class": VALIDATION_FRACTION,
            "seed": SEED,
        },
        "statuses": dict(Counter(record["status"] for record in records)),
        "duplicate_groups": len(duplicate_groups),
        "cross_split_duplicate_groups": sum(group["cross_split"] for group in duplicate_groups),
        "train_max_min_ratio": max(train_counts) / min(train_counts) if train_counts and min(train_counts) else None,
        "image_dimensions_px": {
            key: dict(zip(("min", "median", "max"), np.percentile([entry[key] for entry in kept], [0, 50, 100]).tolist()))
            for key in ("width", "height") if kept
        },
        "missing_classes_by_split": {
            split: [row["class_name"] for row in class_counts if not row[split]] for split in SPLITS
        },
        "issues": issues,
    }
    write_csv(paths.metrics / "dataset_images.csv", AUDIT_FIELDS, records)
    count_fields = ["class_name", *SPLITS, *(f"{split}_candidates" for split in SPLITS)]
    write_csv(paths.metrics / "dataset_class_counts.csv", count_fields, class_counts)
    write_json(paths.metrics / "dataset_duplicates.json", {"groups": duplicate_groups})
    write_csv(paths.metrics / "dataset_validation_repair.csv", ["path", "class_name"], moved)
    write_json(paths.metrics / "dataset_summary.json", summary)
    if issues:
        write_json(paths.manifest, {"issues": issues})
        raise ProjectError(" ".join(issues) + " Xem kết quả kiểm tra trong results/metrics.")

    save_eda_figures(paths, summary, class_counts, kept)
    write_json(paths.manifest, {"dataset_id": summary["dataset_id"], "class_names": class_names, "images": images})
    return summary


def load_manifest(paths: Paths) -> Manifest:
    if not paths.manifest.is_file():
        raise ProjectError("Chưa có manifest. Chạy python -m src.dataset trước.")
    payload = read_json(paths.manifest)
    if payload.get("issues"):
        raise ProjectError("Manifest chưa hợp lệ. Sửa lỗi trong dataset_summary.json rồi chuẩn bị lại dữ liệu.")
    manifest = Manifest(payload["dataset_id"], payload["class_names"], payload["images"])
    if dataset_fingerprint(manifest.images, manifest.class_names) != manifest.dataset_id:
        raise ProjectError("Manifest đã thay đổi. Chạy lại python -m src.dataset.")
    return manifest


def verify_split(paths: Paths, manifest: Manifest, split: str) -> None:
    # Phát hiện ảnh bị thay thế sau EDA. Huấn luyện chỉ kiểm tra train và validation.
    for entry in manifest.images:
        if entry["split"] != split:
            continue
        path = paths.raw / entry["path"]
        if sha256_file(path) != entry["file_hash"]:
            raise ProjectError(f"Ảnh đã thay đổi: {path}. Chạy lại python -m src.dataset.")


def main() -> None:
    args = workspace_parser("Kiểm tra dữ liệu và tạo báo cáo EDA.").parse_args()
    paths = Paths(args.workspace.resolve())
    summary = prepare_dataset(paths)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Đã lưu manifest tại {paths.manifest}")


if __name__ == "__main__":
    run_cli(main)
