"""Appointment scheduling behind a small interface.

MockScheduler is what the demo uses: no network, no PHI, deterministic slots.
Its payloads borrow the shape of FHIR Appointment resources, so a real adapter
(for example one that calls an EHR's find-slots / book operations) can replace
it without touching the app.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Protocol

PCP = {"reference": "Practitioner/demo-pcp", "display": "Dr. Avery Lin, Primary Care (demo)"}


class Scheduler(Protocol):
    def find_slots(self, level: str, now: datetime | None = None) -> list[dict]: ...
    def book(self, slot_id: str, reason: str) -> dict: ...


class MockScheduler:
    def __init__(self) -> None:
        self._offered: dict[str, dict] = {}

    def find_slots(self, level: str, now: datetime | None = None) -> list[dict]:
        now = now or datetime.now()
        hour = now.replace(minute=0, second=0, microsecond=0)
        if level == "URGENT":  # today: next few hours inside clinic hours (8-17)
            starts = [hour + timedelta(hours=h) for h in (1, 2, 3)]
            starts = [s for s in starts if s.date() == now.date() and 8 <= s.hour < 17]
        else:  # SOON / ROUTINE: mornings on the next three weekdays
            starts, day = [], now
            while len(starts) < 3:
                day += timedelta(days=1)
                if day.weekday() < 5:
                    starts.append(day.replace(hour=9 + len(starts), minute=30, second=0, microsecond=0))
        slots: list[dict[str, Any]] = [
            {
                "resourceType": "Appointment",
                "id": f"demo-{start:%Y%m%d%H%M}",
                "status": "proposed",
                "start": start.isoformat(timespec="minutes"),
                "end": (start + timedelta(minutes=20)).isoformat(timespec="minutes"),
                "participant": [{"actor": PCP, "status": "needs-action"}],
            }
            for start in starts
        ]
        self._offered.update({slot["id"]: slot for slot in slots})
        return slots

    def book(self, slot_id: str, reason: str) -> dict:
        slot = self._offered.get(slot_id)
        if slot is None:
            raise KeyError(f"Unknown or expired slot: {slot_id}")
        return {**slot, "status": "booked", "description": reason}


def label(slot: dict) -> str:
    start = datetime.fromisoformat(slot["start"])
    return f"{start:%a %b %d, %I:%M %p} - {slot['participant'][0]['actor']['display']}"
