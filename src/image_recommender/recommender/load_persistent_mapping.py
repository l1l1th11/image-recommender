from pathlib import Path

import numpy as np


def load_persistent_mapping(run_dir: Path):
    run_dir = Path(run_dir)

    id_to_vec_maps = {}

    for feature_dir in run_dir.iterdir():
        if not feature_dir.is_dir():
            continue

        mapping_dir = feature_dir / "mapping"

        ids_path = mapping_dir / "ids.npy"
        vecs_path = mapping_dir / "vecs.npy"

        if not ids_path.exists() or not vecs_path.exists():
            continue

        ids = np.load(ids_path)
        vecs = np.load(vecs_path)

        # 🔥 NEW: build id → index map ONLY (cheap)
        id_to_index = {int(id_): i for i, id_ in enumerate(ids)}

        # store compact structure
        id_to_vec_maps[feature_dir.name] = {
            "ids": ids,
            "vecs": vecs,
            "index": id_to_index,
        }

    return id_to_vec_maps
