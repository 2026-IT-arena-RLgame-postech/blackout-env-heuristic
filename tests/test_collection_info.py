from blackout_env.train.offline_dataset import dataset_has_unity_shaping, write_collection_info


def test_dataset_without_marker_counts_as_shaped(tmp_path):
    assert dataset_has_unity_shaping(tmp_path)


def test_marker_round_trip(tmp_path):
    write_collection_info(tmp_path, unity_shaping=False)
    assert not dataset_has_unity_shaping(tmp_path)
    write_collection_info(tmp_path, unity_shaping=True)
    assert dataset_has_unity_shaping(tmp_path)
