import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_confidential_input_matches_recorded_provenance():
    raw = ROOT / "data/brut.csv"
    provenance = json.loads(
        (ROOT / "data/validation/input_provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["confidential"] is True
    assert provenance["rows"] == 37839
    assert provenance["cows"] == 28
    assert provenance["size_bytes"] == raw.stat().st_size
    assert provenance["sha256"] == hashlib.sha256(raw.read_bytes()).hexdigest()
