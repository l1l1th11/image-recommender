from image_recommender.features.extraction_pipeline import run_extraction
from image_recommender.features.storage import read_validate_shard


def test_samples_extraction(tmp_path):
    # run hsv extraction
    run_extraction(
        feature_type="hsv",
        input_mode="samples",
        run_dir=tmp_path,
        shard_start=None,
        shard_stop=None,
        pilot_path=None,
        db_path=None,
        shard_size=None,
        policy="skip_and_log",
    )

    # run embedding extraction
    run_extraction(
        feature_type="embedding",
        input_mode="samples",
        run_dir=tmp_path,
        shard_start=None,
        shard_stop=None,
        pilot_path=None,
        db_path=None,
        shard_size=None,
        policy="skip_and_log",
    )

    # ensure shard dirs exists
    hsv_dir = tmp_path / "hsv"
    embedding_dir = tmp_path / "embedding"
    assert hsv_dir.exists()
    assert embedding_dir.exists()

    # validate shards
    features_array_hsv, ids_list_hsv = read_validate_shard(
        run_dir=tmp_path, feature_type="hsv", shard_id=0
    )
    features_array_embedding, ids_list_embedding = read_validate_shard(
        run_dir=tmp_path, feature_type="embedding", shard_id=0
    )

    # ensure no empty arrays are returned
    assert len(ids_list_hsv) > 0

    # validate ids and alignment
    assert len(ids_list_hsv) == len(features_array_hsv)
    assert len(ids_list_embedding) == len(features_array_embedding)
    assert ids_list_hsv == ids_list_embedding

    # ensure idempotence
    run_extraction(
        feature_type="hsv",
        input_mode="samples",
        run_dir=tmp_path,
        shard_start=None,
        shard_stop=None,
        pilot_path=None,
        db_path=None,
        shard_size=None,
        policy="skip_and_log",
    )

    run_extraction(
        feature_type="embedding",
        input_mode="samples",
        run_dir=tmp_path,
        shard_start=None,
        shard_stop=None,
        pilot_path=None,
        db_path=None,
        shard_size=None,
        policy="skip_and_log",
    )

    # validate shards
    _, ids_list_hsv_rerun = read_validate_shard(run_dir=tmp_path, feature_type="hsv", shard_id=0)
    _, ids_list_embedding_rerun = read_validate_shard(
        run_dir=tmp_path, feature_type="embedding", shard_id=0
    )

    # validate ids are unchanged
    assert ids_list_hsv == ids_list_hsv_rerun
    assert ids_list_embedding == ids_list_embedding_rerun
