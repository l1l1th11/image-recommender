from pathlib import Path

SAMPLES_DIR = Path("data/samples")

PILOT_IDS_CSV = Path("data/pilot/pilot_1k_ids.csv")

# normalized image extensions (lowercase, no dot)
IMAGE_EXTS: set[str] = {"jpg", "jpeg", "png", "gif", "svg", "webp", "tif", "tiff"}
