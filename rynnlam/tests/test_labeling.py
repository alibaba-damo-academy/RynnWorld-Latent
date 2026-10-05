import json

import numpy as np
import pytest
import torch

import label_latent_dense as labeling
import rynnlam.video as video_module
from rynnlam.video import FrameReader, pair_batches


def write_video(path, n=9):
    import av

    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width, stream.height = 64, 48
        stream.pix_fmt = "yuv420p"
        for i in range(n):
            image = np.full((48, 64, 3), i * 20, dtype=np.uint8)
            for packet in stream.encode(
                av.VideoFrame.from_ndarray(image, format="rgb24")
            ):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


class FakeEncoder:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, images, representation="ktoken"):
        values = torch.from_numpy(images).mean((1, 2, 3, 4))
        return values[:, None, None].expand(-1, 2, 7)


def test_frame_pair_batches(tmp_path):
    video = tmp_path / "video.mp4"
    write_video(video)
    batches = list(pair_batches(FrameReader(video), gap=3, batch_size=4))
    assert [len(b[0]) for b in batches] == [4, 2]
    pairs = np.concatenate([b[1] for b in batches])
    np.testing.assert_array_equal(
        pairs, np.stack([np.arange(6), np.arange(6) + 3], axis=-1)
    )
    assert batches[0][2] == "4x3"
    assert batches[0][0].shape == (4, 2, 238, 322, 3)
    with pytest.raises(ValueError, match="incomplete interval"):
        list(pair_batches(FrameReader(video), end=20))
    with pytest.raises(ValueError, match="gap"):
        list(pair_batches(FrameReader(video), gap=9))


def test_label_cli_resume_and_rebuild(tmp_path, monkeypatch):
    video = tmp_path / "video.mp4"
    write_video(video)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"mock-checkpoint")
    output = tmp_path / "labels"
    monkeypatch.setattr(labeling, "RynnLAMEncoder", FakeEncoder)
    args = [
        "--video",
        str(video),
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(output),
        "--gap",
        "3",
        "--pair-stride",
        "1",
        "--device",
        "cpu",
        "--precision",
        "fp32",
        "--batch-size",
        "2",
    ]
    assert labeling.main(args) == 0
    result = output / "latents/videos/video/head/latent.npz"
    with np.load(result, allow_pickle=False) as data:
        assert data["latent_action"].shape == (6, 14)
        assert np.isfinite(data["latent_action"]).all()
        meta = json.loads(str(data["meta"].item()))
        assert meta["code_shape"] == [2, 7]
        assert meta["fps"] == 30
    index = output / "index_shard0000.json"
    assert json.loads(index.read_text())[0]["views"]["head"]["num_latents"] == 6

    def no_model(*args, **kwargs):
        raise AssertionError("Resume and rebuilding should not load a model")

    monkeypatch.setattr(labeling, "RynnLAMEncoder", no_model)
    assert labeling.main(args) == 0
    index.unlink()
    assert labeling.main(args + ["--rebuild-index"]) == 0
    checkpoint.write_bytes(b"changed-checkpoint")
    assert labeling.main(args) == 1
    assert json.loads((output / "failed_shard0000.json").read_text())
    assert result.is_file()


def test_unsafe_identifiers():
    for value in ["../escape", "/absolute", "", "..", "bad\\path"]:
        with pytest.raises(ValueError):
            labeling.safe_name(value, nested=True)


def test_streaming_validation_rejects_bad_values_and_pairs(tmp_path):
    path = tmp_path / "labels.npz"
    protocol = {"schema_version": 2, "gap": 5, "pair_stride": 1}
    meta = {
        "protocol": protocol,
        "num_latents": 3,
        "num_frames": 8,
        "unpaired_tail_frames": 0,
        "code_shape": [2, 7],
    }
    latent = np.zeros((3, 14), np.float16)
    pairs = np.stack([np.arange(3), np.arange(3) + 5], axis=-1).astype(np.int32)
    np.savez_compressed(
        path, latent_action=latent, pair_indices=pairs, meta=json.dumps(meta)
    )
    assert labeling.output_metadata(path, protocol) == meta
    latent[1, 2] = np.nan
    np.savez_compressed(
        path, latent_action=latent, pair_indices=pairs, meta=json.dumps(meta)
    )
    with pytest.raises(ValueError, match="Non-finite"):
        labeling.output_metadata(path, protocol)
    latent[1, 2] = 0
    pairs[1, 0] = 10
    np.savez_compressed(
        path, latent_action=latent, pair_indices=pairs, meta=json.dumps(meta)
    )
    with pytest.raises(ValueError, match="alignment"):
        labeling.output_metadata(path, protocol)


def test_multiple_views_rebuild_requires_each_label(tmp_path, monkeypatch):
    video = tmp_path / "video.mp4"
    write_video(video)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"mock-checkpoint")
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps(
            [
                {
                    "dataset": "sample",
                    "episode_id": "episode/1",
                    "views": {
                        "head": {"video_path": "video.mp4"},
                        "wrist_left": {"video_path": "video.mp4"},
                    },
                }
            ]
        )
    )
    output = tmp_path / "labels"
    monkeypatch.setattr(labeling, "RynnLAMEncoder", FakeEncoder)
    args = [
        "--metadata",
        str(metadata),
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(output),
    ]
    assert labeling.main(args) == 0
    (output / "latents/sample/episode/1/wrist_left/latent.npz").unlink()
    assert labeling.main(args + ["--rebuild-index"]) == 1
    assert labeling.main(args) == 0
    assert set(
        json.loads((output / "index_shard0000.json").read_text())[0]["views"]
    ) == {"head", "wrist_left"}


class DummyReader:
    """Distinct constant RGB frames keep temporal checks exact and inexpensive."""

    def __init__(self, n):
        self.n = n
        self.num_frames = None
        self.fps = 30.0

    def frames(self, start=0, end=None):
        stop = self.n if end is None else min(end, self.n)
        count = 0
        for index in range(start, stop):
            count += 1
            yield np.full((3, 4, 3), index, dtype=np.uint8)
        self.num_frames = count
        if not count or (end is not None and count != end - start):
            raise ValueError("incomplete interval")


@pytest.fixture
def tiny_buckets(monkeypatch):
    monkeypatch.setitem(video_module.BUCKETS, "4x3", (3, 4))


@pytest.mark.parametrize(
    "n,expected_count", [(4, 0), (5, 1), (8, 1), (9, 2), (10, 2), (97, 24)]
)
def test_sparse_pair_lengths(n, expected_count, tiny_buckets):
    reader = DummyReader(n)
    iterator = pair_batches(reader, gap=4, pair_stride=4, batch_size=5)
    if expected_count == 0:
        with pytest.raises(ValueError, match="gap"):
            list(iterator)
        assert reader.num_frames == n
        return
    batches = list(iterator)
    pairs = np.concatenate([batch[1] for batch in batches])
    starts = np.arange(expected_count) * 4
    np.testing.assert_array_equal(pairs, np.column_stack((starts, starts + 4)))
    images = np.concatenate([batch[0] for batch in batches])
    np.testing.assert_allclose(images[:, :, 0, 0, 0], pairs / 255.0)
    assert reader.num_frames == n
    assert pairs[-1, 1] < n
    assert all(len(batch[0]) == 5 for batch in batches[:-1])
    assert len(batches[-1][0]) == (expected_count - 1) % 5 + 1


@pytest.mark.parametrize("gap", [4, 5])
def test_sparse_interval_indices_and_batch_boundaries(gap, tiny_buckets):
    reader = DummyReader(30)
    batches = list(
        pair_batches(reader, gap=gap, pair_stride=4, batch_size=2, start=3, end=21)
    )
    starts = np.arange(0, 18 - gap, 4)
    pairs = np.concatenate([batch[1] for batch in batches])
    np.testing.assert_array_equal(pairs, np.column_stack((starts, starts + gap)))
    images = np.concatenate([batch[0] for batch in batches])
    np.testing.assert_allclose(images[:, :, 0, 0, 0], (pairs + 3) / 255.0)
    assert [len(batch[0]) for batch in batches] == [2, 2]
    assert reader.num_frames == 18


@pytest.mark.parametrize("gap", [4, 5])
def test_sparse_tokens_match_dense_subset(gap, tiny_buckets):
    encoder = FakeEncoder()

    def encode(stride, batch_size):
        batches = list(
            pair_batches(
                DummyReader(23), gap=gap, pair_stride=stride, batch_size=batch_size
            )
        )
        return (
            np.concatenate([encoder(batch[0]).numpy() for batch in batches]),
            np.concatenate([batch[1] for batch in batches]),
        )

    dense_tokens, dense_pairs = encode(1, 3)
    sparse_tokens, sparse_pairs = encode(4, 2)
    np.testing.assert_array_equal(sparse_pairs, dense_pairs[::4])
    np.testing.assert_array_equal(sparse_tokens, dense_tokens[::4])
    assert np.all(
        sparse_tokens[0] > 0
    ), "The first label must encode (0, gap), not padding"


@pytest.mark.parametrize("gap,n", [(4, 10), (4, 97), (5, 10)])
def test_sparse_resizes_only_needed_endpoints(gap, n, tiny_buckets, monkeypatch):
    resized = []
    original = video_module.resize_crop

    def track(frame, target_hw):
        resized.append(int(frame[0, 0, 0]))
        return original(frame, target_hw)

    monkeypatch.setattr(video_module, "resize_crop", track)
    batches = list(pair_batches(DummyReader(n), gap=gap, pair_stride=4, batch_size=2))
    pairs = np.concatenate([batch[1] for batch in batches])
    expected = sorted(set(pairs.ravel().tolist()))
    assert sorted(resized) == expected


@pytest.mark.parametrize("start,end,expected", [(0, None, 10), (3, None, 7), (3, 9, 6)])
def test_frame_reader_reports_exact_interval_length(tmp_path, start, end, expected):
    video = tmp_path / "video.mp4"
    write_video(video, n=10)
    reader = FrameReader(video)
    frames = list(reader.frames(start=start, end=end))
    assert len(frames) == expected
    assert reader.num_frames == expected


@pytest.mark.parametrize(
    "n,expected_count,tail",
    [(4, 0, None), (5, 1, 0), (8, 1, 3), (9, 2, 0), (10, 2, 1), (97, 24, 0)],
)
def test_label_video_preserves_true_frame_count(
    tmp_path, monkeypatch, tiny_buckets, n, expected_count, tail
):
    monkeypatch.setattr(labeling, "FrameReader", lambda *args, **kwargs: DummyReader(n))
    protocol = {
        "schema_version": 2,
        "gap": 4,
        "pair_stride": 4,
        "bucket": "auto",
        "start_frame": 0,
        "end_frame": None,
        "representation": "ktoken",
    }
    destination = tmp_path / "latent.npz"
    if not expected_count:
        with pytest.raises(ValueError, match="gap"):
            labeling.label_video(
                FakeEncoder(), "unused", destination, protocol, batch_size=5
            )
        assert not destination.exists()
        return
    meta = labeling.label_video(
        FakeEncoder(), "unused", destination, protocol, batch_size=5
    )
    assert meta["num_frames"] == n
    assert meta["num_latents"] == expected_count
    assert meta["unpaired_tail_frames"] == tail
    assert labeling.output_metadata(destination, protocol) == meta
    with np.load(destination, allow_pickle=False) as data:
        starts = np.arange(expected_count) * 4
        np.testing.assert_array_equal(
            data["pair_indices"], np.column_stack((starts, starts + 4))
        )
        assert data["latent_action"].shape == (expected_count, 14)


@pytest.mark.parametrize("stride_option", ["--pair-stride", "--stride"])
def test_cli_defaults_and_stride_resume_refusal(tmp_path, monkeypatch, stride_option):
    video = tmp_path / "video.mp4"
    write_video(video, n=10)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"mock-checkpoint")
    output = tmp_path / "labels"
    monkeypatch.setattr(labeling, "RynnLAMEncoder", FakeEncoder)
    args = [
        "--video",
        str(video),
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(output),
        "--device",
        "cpu",
        "--precision",
        "fp32",
    ]
    assert labeling.main(args) == 0
    result = output / "latents/videos/video/head/latent.npz"
    original = result.read_bytes()
    with np.load(result, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["pair_indices"], [[0, 4], [4, 8]])
        assert data["latent_action"].shape == (2, 14)
        meta = json.loads(str(data["meta"].item()))
    assert meta["protocol"]["schema_version"] == 2
    assert meta["protocol"]["gap"] == meta["protocol"]["pair_stride"] == 4
    assert meta["num_frames"] == 10
    assert meta["unpaired_tail_frames"] == 1
    view = json.loads((output / "index_shard0000.json").read_text())[0]["views"]["head"]
    assert view["num_frames"] == 10
    assert view["num_latents"] == 2

    def no_model(*args, **kwargs):
        raise AssertionError(
            "Compatible resume or incompatible protocol must not load a model"
        )

    monkeypatch.setattr(labeling, "RynnLAMEncoder", no_model)
    assert labeling.main(args + [stride_option, "4"]) == 0
    assert labeling.main(args + [stride_option, "1"]) == 1
    errors = json.loads((output / "failed_shard0000.json").read_text())
    assert len(errors) == 1
    assert "protocol" in errors[0]["error"].lower()
    assert result.read_bytes() == original


@pytest.mark.parametrize("stride", [0, -1])
def test_pair_batches_reject_invalid_stride(stride):
    with pytest.raises(ValueError, match="stride"):
        list(pair_batches(DummyReader(10), gap=4, pair_stride=stride))


@pytest.mark.parametrize("option", ["--pair-stride", "--stride"])
def test_cli_rejects_zero_stride(tmp_path, capsys, option):
    with pytest.raises(SystemExit) as error:
        labeling.main(
            [
                "--video",
                str(tmp_path / "missing.mp4"),
                "--checkpoint",
                str(tmp_path / "missing.pt"),
                "--output-dir",
                str(tmp_path / "labels"),
                option,
                "0",
            ]
        )
    assert error.value.code == 2
    message = capsys.readouterr().err.lower()
    assert "stride" in message
    assert "unrecognized arguments" not in message
    assert not (tmp_path / "labels").exists()


@pytest.mark.parametrize("gap", [4, 5])
def test_streaming_validation_checks_sparse_alignment(tmp_path, gap):
    path = tmp_path / "labels.npz"
    protocol = {"schema_version": 2, "gap": gap, "pair_stride": 4}
    meta = {
        "protocol": protocol,
        "num_latents": 3,
        "num_frames": gap + 11,
        "unpaired_tail_frames": 2,
        "code_shape": [2, 7],
    }
    latent = np.zeros((3, 14), np.float16)
    starts = np.arange(3, dtype=np.int32) * 4
    pairs = np.column_stack((starts, starts + gap))
    np.savez_compressed(
        path, latent_action=latent, pair_indices=pairs, meta=json.dumps(meta)
    )
    assert labeling.output_metadata(path, protocol) == meta
    dense_starts = np.arange(3, dtype=np.int32)
    bad_pairs = [
        np.column_stack((dense_starts, dense_starts + gap)),
        pairs + 1,
        pairs[:, ::-1].copy(),
    ]
    wrong_target = pairs.copy()
    wrong_target[-1, 1] += 1
    bad_pairs.append(wrong_target)
    for invalid in bad_pairs:
        np.savez_compressed(
            path, latent_action=latent, pair_indices=invalid, meta=json.dumps(meta)
        )
        with pytest.raises(ValueError, match="alignment"):
            labeling.output_metadata(path, protocol)
