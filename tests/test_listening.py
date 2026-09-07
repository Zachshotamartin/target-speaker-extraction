import json

import pytest
from pydantic import ValidationError

from tse.listening import ListeningRating, save_rating, summarize_listening
from tse.utils import atomic_json


def test_only_submitted_ratings_count_and_repeat_submission_replaces(tmp_path):
    directory = tmp_path / "test-study"
    atomic_json(
        directory / "key.json",
        {"trials": [{"id": "trial-01", "models": {"A": "model-left", "B": "model-right"}}]},
    )
    assert summarize_listening(tmp_path, "test-study")["participants"] == 0
    scores = {"competing_speech": 2, "target_damage": 3, "static": 4}
    rating = ListeningRating(participant="unit-test", trial="trial-01", A=scores, B=scores)
    assert save_rating(tmp_path, "test-study", rating)["completed_trials"] == 1
    rating.A.static = 1
    save_rating(tmp_path, "test-study", rating)
    report = summarize_listening(tmp_path, "test-study")
    assert report["participants"] == 1
    assert report["models"]["model-left"]["ratings"] == 1
    assert report["models"]["model-left"]["static"] == 1
    with pytest.raises(ValueError, match="Invalid study"):
        save_rating(tmp_path, "../escape", rating)
    rating.trial = "unknown"
    with pytest.raises(ValueError, match="Unknown listening"):
        save_rating(tmp_path, "test-study", rating)
    assert len(json.loads((directory / "ratings/unit-test.json").read_text())["ratings"]) == 1


def test_rating_rejects_missing_criteria_and_out_of_range_scores():
    with pytest.raises(ValidationError):
        ListeningRating(participant="unit-test", trial="trial", A={"static": 0}, B={"static": 6})
