"""Audita los manifiestos y guarda hashes/evidencia sin modificarlos.

Uso: ``python scripts/audit_splits.py`` (desde la raíz del repo).
Escribe ``data/splits/SHA256SUMS`` y ``data/splits/partition_audit.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from leafthealthvision.plantvillage import audit_manifests, load_splits, validate_manifest_splits

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, default=ROOT / "data" / "splits")
    parser.add_argument("--color-dir", type=Path, default=ROOT / "data" / "plantvillage" / "raw" / "color")
    args = parser.parse_args()

    manifest_dir = args.manifest_dir.resolve()
    audit = validate_manifest_splits(manifest_dir)
    splits = load_splits(manifest_dir, args.color_dir.resolve())
    audit["resolved_counts"] = {name: len(records.paths) for name, records in splits.items()}
    audit["unresolved_counts"] = {name: len(records.unresolved_image_ids) for name, records in splits.items()}
    audit["resolved_total"] = sum(audit["resolved_counts"].values())
    audit["unresolved_examples"] = [
        {"split": name, "image_id": image_id}
        for name, records in splits.items()
        for image_id in records.unresolved_image_ids[:10]
    ]
    if manifest_dir.is_relative_to(ROOT):
        audit["manifest_dir"] = manifest_dir.relative_to(ROOT).as_posix()

    sums = "".join(f"{digest}  {name}.csv\n" for name, digest in audit["sha256"].items())
    with (manifest_dir / "SHA256SUMS").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(sums)
    (manifest_dir / "partition_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(
        {
            "counts": audit["counts"],
            "total_rows": audit["total_rows"],
            "classes_per_split": {name: len(classes) for name, classes in audit["classes_by_split"].items()},
            "same_classes": audit["same_classes"],
            "rows_without_leaf": audit["rows_without_leaf"],
            "leaf_groups_shared_across_splits": audit["leaf_groups_shared_across_splits"],
            "duplicate_manifest_rows": audit["duplicate_manifest_rows"],
            "split_field_mismatches": audit["split_field_mismatches"],
            "resolved_counts": audit["resolved_counts"],
            "unresolved_counts": audit["unresolved_counts"],
            "sha256": audit["sha256"],
        },
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
