"""
alerts.py  —  evaluate users' rent / affordability alerts (Milestone 5).

An alert watches one (area, property type) and fires when the model's current
estimate crosses a threshold:
    metric = 'rent'   -> predicted monthly rent (£)
    metric = 'ratio'  -> rent-to-income ratio (0.30 = the affordability benchmark)
    op     = 'above' | 'below'

Edge-triggered: a notification is written only when the condition turns TRUE
(it was false or never checked), so a user is not re-notified every run while it
stays true. Run after each data/model refresh (monthly workflow):

    python -m src.api.alerts            # evaluates every active alert

Delivery is pluggable: notifications are always stored (GET /v1/me/notifications);
an optional Notifier (e.g. SMTP via env vars) can also push them out.
"""

from __future__ import annotations

import logging
import os
import smtplib
from email.message import EmailMessage

from src.api.db import Database, now_iso

log = logging.getLogger("rentwise.alerts")


class LogNotifier:
    name = "log"

    def send(self, email: str, subject: str, body: str) -> bool:
        log.info("notify %s: %s", email, subject)
        return True


class SMTPNotifier:
    """Sends email when RENTWISE_SMTP_HOST (+ _PORT, _USER, _PASSWORD, _FROM) are set."""
    name = "email"

    def __init__(self):
        self.host = os.environ["RENTWISE_SMTP_HOST"]
        self.port = int(os.environ.get("RENTWISE_SMTP_PORT", "587"))
        self.user = os.environ.get("RENTWISE_SMTP_USER")
        self.password = os.environ.get("RENTWISE_SMTP_PASSWORD")
        self.sender = os.environ.get("RENTWISE_SMTP_FROM", "alerts@rentwise.local")

    def send(self, email: str, subject: str, body: str) -> bool:
        msg = EmailMessage()
        msg["From"], msg["To"], msg["Subject"] = self.sender, email, subject
        msg.set_content(body)
        try:
            with smtplib.SMTP(self.host, self.port, timeout=15) as s:
                s.starttls()
                if self.user:
                    s.login(self.user, self.password or "")
                s.send_message(msg)
            return True
        except Exception as e:  # noqa: BLE001 — delivery failure must not stop evaluation
            log.warning("email to %s failed: %s", email, e)
            return False


def default_notifier():
    return SMTPNotifier() if os.environ.get("RENTWISE_SMTP_HOST") else LogNotifier()


def current_value(predictor, la_code: str, property_type: str, metric: str) -> float | None:
    area = predictor.area_by_code(la_code)
    if area is None:
        return None
    r = predictor.predict(area["la_name"], property_type)
    return float(r["rent"]) if metric == "rent" else float(r["ratio"])


def describe(alert: dict, area_name: str, value: float) -> str:
    if alert["metric"] == "rent":
        return (f"{alert['property_type']} rent in {area_name} is now £{value:,.0f}/month — "
                f"{alert['op']} your £{alert['threshold']:,.0f} threshold.")
    return (f"Rent-to-income for a {alert['property_type'].lower()} in {area_name} is now "
            f"{value * 100:.0f}% — {alert['op']} your {alert['threshold'] * 100:.0f}% threshold.")


def evaluate_all(db: Database, predictor, notifier=None) -> dict:
    notifier = notifier or LogNotifier()
    alerts = db.all("SELECT a.*, u.email FROM alerts a JOIN users u ON u.id = a.user_id WHERE a.active = 1")
    fired = checked = skipped = 0
    for a in alerts:
        value = current_value(predictor, a["la_code"], a["property_type"], a["metric"])
        if value is None:
            skipped += 1
            continue
        checked += 1
        state = value > a["threshold"] if a["op"] == "above" else value < a["threshold"]
        if state and not a["last_state"]:
            area_name = predictor.area_by_code(a["la_code"])["la_name"]
            msg = describe(a, area_name, value)
            ok = notifier.send(a["email"], "RentWise alert", msg)
            db.run("INSERT INTO notifications (user_id, alert_id, message, value, created_at, delivered_via) "
                   "VALUES (?, ?, ?, ?, ?, ?)",
                   (a["user_id"], a["id"], msg, value, now_iso(), notifier.name if ok else "stored"))
            fired += 1
        db.run("UPDATE alerts SET last_value = ?, last_state = ?, last_checked_at = ? WHERE id = ?",
               (value, int(state), now_iso(), a["id"]))
    summary = {"alerts": len(alerts), "checked": checked, "fired": fired, "skipped_unknown_area": skipped}
    log.info("alerts evaluated: %s", summary)
    return summary


if __name__ == "__main__":  # pragma: no cover
    import sys
    from pathlib import Path
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.api.app import db_path_from_env
    from src.models.predict import RentWisePredictor
    print(evaluate_all(Database(db_path_from_env()), RentWisePredictor(), default_notifier()))
