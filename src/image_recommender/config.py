from pathlib import Path

SAMPLES_DIR = Path("data/samples")

PILOT_IDS_CSV = Path("data/pilot/pilot_1k_ids.csv")

DB_PATH = Path("data/metadata.db")

DEFAULT_PILOT_SHARD_SIZE = 200

DEFAULT_FULL_SHARD_SIZE = 5000

ANNOY_DEFAULT_N_TREES = 200

ANNOY_DEFAULT_METRIC = "angular"

ANNOY_DEFAULT_SEARCH_K = 5_000_000

DR_SEED = 42

PHASH_SIZE = 32

PHASH_DIM = 64
