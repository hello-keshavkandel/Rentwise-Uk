"""make demo seeds a working, separate demo account."""
import pytest

import config

pytestmark = pytest.mark.skipif(not (config.MODELS_DIR / "rent_model_meta.json").exists(),
                                reason="model not built")


def test_seed_creates_usable_demo_account(tmp_path):
    from scripts.demo import DEMO_EMAIL, DEMO_PASSWORD, seed
    from src.api import security as sec
    from src.api.db import Database
    db_path = tmp_path / "demo.sqlite3"
    info = seed(db_path)
    db = Database(db_path)
    u = db.one("SELECT id, pw_hash FROM users WHERE email = ?", (DEMO_EMAIL,))
    assert u and sec.verify_password(DEMO_PASSWORD, u["pw_hash"])
    assert db.one("SELECT user_id FROM api_keys WHERE key_hash = ?", (sec.sha256(info["api_key"]),))["user_id"] == u["id"]
    assert db.one("SELECT COUNT(*) c FROM saved_areas")["c"] == 1
    assert info["alerts"]["checked"] == 1
    assert db.one("SELECT COUNT(*) c FROM notifications")["c"] == info["alerts"]["fired"]
