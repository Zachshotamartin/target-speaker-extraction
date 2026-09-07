"""Local blind listening forms; only user-submitted ratings count as listening evidence."""

import json
import threading
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from tse.utils import atomic_json

_RATING_LOCK = threading.Lock()


class AudioRatings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    competing_speech: int = Field(ge=1, le=5)
    target_damage: int = Field(ge=1, le=5)
    static: int = Field(ge=1, le=5)


class ListeningRating(BaseModel):
    model_config = ConfigDict(extra="forbid")
    participant: str = Field(pattern=r"^[a-zA-Z0-9-]{1,64}$")
    trial: str = Field(pattern=r"^[a-zA-Z0-9-]{1,64}$")
    A: AudioRatings
    B: AudioRatings
    comment: str = Field(default="", max_length=1000)


def save_rating(root: Path, study: str, rating: ListeningRating):
    if not study or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in study):
        raise ValueError("Invalid study identifier")
    directory = root / study
    key_path = directory / "key.json"
    if not key_path.is_file():
        raise FileNotFoundError("Listening study is not available")
    key = json.loads(key_path.read_text())
    if rating.trial not in {t["id"] for t in key["trials"]}:
        raise ValueError("Unknown listening trial")
    output = directory / "ratings" / f"{rating.participant}.json"
    with _RATING_LOCK:
        current = (
            json.loads(output.read_text())
            if output.exists()
            else {"study": study, "participant": rating.participant, "ratings": {}}
        )
        current["ratings"][rating.trial] = {
            **rating.model_dump(exclude={"participant", "trial"}),
            "submitted_at_utc": datetime.now(UTC).isoformat(),
        }
        atomic_json(output, current)
    return {
        "saved": True,
        "completed_trials": len(current["ratings"]),
        "total_trials": len(key["trials"]),
    }


def summarize_listening(root: Path, study: str) -> dict:
    directory = root / study
    key = json.loads((directory / "key.json").read_text())
    trials = {t["id"]: t for t in key["trials"]}
    values = {}
    participants = []
    for path in sorted((directory / "ratings").glob("*.json")):
        payload = json.loads(path.read_text())
        participants.append(payload["participant"])
        for trial_id, scores in payload["ratings"].items():
            for label in ("A", "B"):
                model = trials[trial_id]["models"][label]
                values.setdefault(model, []).append(scores[label])
    return {
        "study": study,
        "status": "Human ratings received"
        if participants
        else "Awaiting human listening; no subjective result claimed",
        "participants": len(participants),
        "models": {
            model: {
                "ratings": len(rows),
                **{
                    criterion: sum(r[criterion] for r in rows) / len(rows)
                    for criterion in ("competing_speech", "target_damage", "static")
                },
            }
            for model, rows in values.items()
        },
        "interpretation": "Ordinal 1..5 severity ratings; lower is better. This convenience listening panel is not a population-level perceptual study.",
    }
