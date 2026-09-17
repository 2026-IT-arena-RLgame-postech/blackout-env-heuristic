from blackout_env.train.offline_dataset import dataset_has_unity_shaping, write_collection_info


def test_dataset_without_marker_counts_as_shaped(tmp_path):
    assert dataset_has_unity_shaping(tmp_path)


def test_marker_round_trip(tmp_path):
    write_collection_info(tmp_path, unity_shaping=False)
    assert not dataset_has_unity_shaping(tmp_path)
    write_collection_info(tmp_path, unity_shaping=True)
    assert dataset_has_unity_shaping(tmp_path)


def test_merge_shards_concatenates_in_order(tmp_path):
    import numpy as np

    from blackout_env.train.offline_dataset import FIELDS, merge_shards, npz_member_memmap

    shards = []
    for k, n in enumerate((3, 5)):
        arrays = {f: np.full((n, 2), k + i, dtype=np.float32) for i, f in enumerate(FIELDS)}
        path = tmp_path / f"s{k}.npz"
        np.savez(path, **arrays)
        shards.append(path)
    assert merge_shards(shards, tmp_path / "out.npz") == 8
    reward = np.asarray(npz_member_memmap(tmp_path / "out.npz", "reward"))
    i = FIELDS.index("reward")
    assert (reward[:3] == i).all() and (reward[3:] == 1 + i).all()
