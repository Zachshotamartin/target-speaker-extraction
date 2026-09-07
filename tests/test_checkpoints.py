import pytest
import torch

from tse.checkpoints import average
from tse.engine import save_checkpoint
from tse.utils import sha256


def test_average_preserves_source_states_and_rejects_different_training_labels(tmp_path):
    paths = [tmp_path / f"source-{i}.pt" for i in (1, 2)]
    for i, path in enumerate(paths):
        save_checkpoint(
            path,
            {
                "format_version": 1,
                "config": {"experiment": "same"},
                "step": i + 1,
                "speaker_classes": 2,
                "manifest_sha256": "source",
                "dev_cases_sha256": "dev",
                "provenance": {"train_speakers": ["1", "2"]},
                "model": {"weight": torch.tensor([1.0, 3.0]) * (i + 1)},
                "optimizer": {"state": "must not survive averaging"},
            },
        )
    before = [sha256(path) for path in paths]
    output = tmp_path / "average.pt"
    average(paths, output)
    payload = torch.load(output, weights_only=True)
    torch.testing.assert_close(payload["model"]["weight"], torch.tensor([1.5, 4.5]))
    assert payload["step"] == 2 and payload["best_score"] is None
    assert "optimizer" not in payload
    assert before == [sha256(path) for path in paths]
    with pytest.raises(FileExistsError):
        average(paths, output)
    changed = torch.load(paths[1], weights_only=True)
    changed["provenance"]["train_speakers"] = ["3", "4"]
    save_checkpoint(paths[1], changed)
    with pytest.raises(ValueError, match="training labels"):
        average(paths, tmp_path / "invalid.pt")
