"""SQLite persistence. One database per agent home. Audit log is append-only.

Tables: episodes, claims (current), claim_versions (history), questions, reflections,
kv (affect/mental state), audit, snapshots.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from .schemas import (AffectState, Episode, MentalState, Question, ReflectionRecord,
                      SelfClaim, now)

SCHEMA_VERSION = 1

DDL = """
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS episodes (id TEXT PRIMARY KEY, ts TEXT, importance REAL, condition TEXT, doc TEXT);
CREATE TABLE IF NOT EXISTS claims (id TEXT PRIMARY KEY, status TEXT, version INTEGER, doc TEXT);
CREATE TABLE IF NOT EXISTS claim_versions (claim_id TEXT, version INTEGER, ts TEXT, doc TEXT,
    PRIMARY KEY (claim_id, version));
CREATE TABLE IF NOT EXISTS questions (id TEXT PRIMARY KEY, status TEXT, doc TEXT);
CREATE TABLE IF NOT EXISTS reflections (id TEXT PRIMARY KEY, ts TEXT, doc TEXT);
CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS audit (seq INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, kind TEXT, doc TEXT);
CREATE TABLE IF NOT EXISTS snapshots (name TEXT PRIMARY KEY, ts TEXT, doc TEXT);
CREATE TABLE IF NOT EXISTS evidence_class (claim_id TEXT, episode_id TEXT, polarity TEXT, note TEXT, PRIMARY KEY (claim_id, episode_id));
"""


def _dump(model) -> str:
    return model.model_dump_json()


class Store:
    def __init__(self, path: Path):
        self.path = path
        self.conn = sqlite3.connect(str(path), check_same_thread=False)  # one writer thread per home; lives run in parallel threads
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(DDL)
        self.conn.execute("INSERT OR IGNORE INTO meta VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
        self.conn.commit()

    # ---- audit (append-only) ----
    def audit(self, kind: str, **doc) -> None:
        self.conn.execute("INSERT INTO audit (ts, kind, doc) VALUES (?,?,?)",
                          (now().isoformat(), kind, json.dumps(doc, default=str)))
        self.conn.commit()

    def audit_tail(self, n: int = 50) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM audit ORDER BY seq DESC LIMIT ?", (n,)).fetchall()
        return [{"seq": r["seq"], "ts": r["ts"], "kind": r["kind"], **json.loads(r["doc"])} for r in rows]

    # ---- episodes ----
    def add_episode(self, ep: Episode) -> None:
        self.conn.execute("INSERT INTO episodes VALUES (?,?,?,?,?)",
                          (ep.id, ep.ts.isoformat(), ep.importance, ep.condition, _dump(ep)))
        self.conn.commit()
        self.audit("episode_added", id=ep.id, event=ep.event[:80])

    def update_episode(self, ep: Episode) -> None:
        self.conn.execute("UPDATE episodes SET importance=?, doc=? WHERE id=?", (ep.importance, _dump(ep), ep.id))
        self.conn.commit()

    def get_episode(self, eid: str) -> Optional[Episode]:
        r = self.conn.execute("SELECT doc FROM episodes WHERE id=?", (eid,)).fetchone()
        return Episode.model_validate_json(r["doc"]) if r else None

    def episodes(self, limit: Optional[int] = None, since: Optional[datetime] = None) -> list[Episode]:
        q, args = "SELECT doc FROM episodes", []
        if since:
            q += " WHERE ts >= ?"; args.append(since.isoformat())
        q += " ORDER BY ts DESC"
        if limit:
            q += " LIMIT ?"; args.append(limit)
        return [Episode.model_validate_json(r["doc"]) for r in self.conn.execute(q, args)]

    def episode_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]

    # ---- claims ----
    def upsert_claim(self, c: SelfClaim, reason: str = "") -> None:
        c.updated_at = now()
        self.conn.execute("INSERT OR REPLACE INTO claims VALUES (?,?,?,?)", (c.id, c.status, c.version, _dump(c)))
        self.conn.execute("INSERT OR REPLACE INTO claim_versions VALUES (?,?,?,?)",
                          (c.id, c.version, c.updated_at.isoformat(), _dump(c)))
        self.conn.commit()
        self.audit("claim_upsert", id=c.id, version=c.version, confidence=round(c.confidence, 3),
                   status=c.status, reason=reason)

    def get_claim(self, cid: str) -> Optional[SelfClaim]:
        r = self.conn.execute("SELECT doc FROM claims WHERE id=?", (cid,)).fetchone()
        return SelfClaim.model_validate_json(r["doc"]) if r else None

    def claims(self, include_inactive: bool = False) -> list[SelfClaim]:
        rows = self.conn.execute("SELECT doc FROM claims").fetchall()
        out = [SelfClaim.model_validate_json(r["doc"]) for r in rows]
        if not include_inactive:
            out = [c for c in out if c.status in ("hypothesis", "belief")]
        return sorted(out, key=lambda c: -c.confidence)

    def claim_versions(self, cid: str) -> list[SelfClaim]:
        rows = self.conn.execute("SELECT doc FROM claim_versions WHERE claim_id=? ORDER BY version", (cid,))
        return [SelfClaim.model_validate_json(r["doc"]) for r in rows]

    def rollback_claim(self, cid: str, to_version: int) -> SelfClaim:
        r = self.conn.execute("SELECT doc FROM claim_versions WHERE claim_id=? AND version=?",
                              (cid, to_version)).fetchone()
        if not r:
            raise KeyError(f"{cid} v{to_version} not found")
        old = SelfClaim.model_validate_json(r["doc"])
        cur = self.get_claim(cid)
        old.version = (cur.version if cur else to_version) + 1   # rollback is itself a new version
        self.upsert_claim(old, reason=f"rollback to v{to_version}")
        return old

    # ---- questions ----
    def upsert_question(self, q: Question) -> None:
        self.conn.execute("INSERT OR REPLACE INTO questions VALUES (?,?,?)", (q.id, q.status, _dump(q)))
        self.conn.commit()

    def questions(self, status: Optional[str] = "open") -> list[Question]:
        q = "SELECT doc FROM questions" + (" WHERE status=?" if status else "")
        rows = self.conn.execute(q, (status,) if status else ()).fetchall()
        return sorted([Question.model_validate_json(r["doc"]) for r in rows], key=lambda x: -x.curiosity)

    # ---- reflections ----
    def add_reflection(self, r: ReflectionRecord) -> None:
        self.conn.execute("INSERT INTO reflections VALUES (?,?,?)", (r.id, r.ts.isoformat(), _dump(r)))
        self.conn.commit()
        self.audit("reflection", id=r.id, verdict=r.verdict, claim_id=r.claim_id, trigger=r.trigger)

    def reflections(self, limit: int = 20) -> list[ReflectionRecord]:
        rows = self.conn.execute("SELECT doc FROM reflections ORDER BY ts DESC LIMIT ?", (limit,))
        return [ReflectionRecord.model_validate_json(r["doc"]) for r in rows]

    def get_reflection(self, rid: str) -> Optional[ReflectionRecord]:
        r = self.conn.execute("SELECT doc FROM reflections WHERE id=?", (rid,)).fetchone()
        return ReflectionRecord.model_validate_json(r["doc"]) if r else None

    # ---- kv state ----
    def _kv_get(self, k: str) -> Optional[str]:
        r = self.conn.execute("SELECT v FROM kv WHERE k=?", (k,)).fetchone()
        return r["v"] if r else None

    def _kv_set(self, k: str, v: str) -> None:
        self.conn.execute("INSERT OR REPLACE INTO kv VALUES (?,?)", (k, v))
        self.conn.commit()

    def load_affect(self) -> AffectState:
        v = self._kv_get("affect")
        return AffectState.model_validate_json(v) if v else AffectState()

    def save_affect(self, a: AffectState) -> None:
        self._kv_set("affect", _dump(a))

    def load_mental(self) -> MentalState:
        v = self._kv_get("mental")
        return MentalState.model_validate_json(v) if v else MentalState()

    def save_mental(self, m: MentalState) -> None:
        self._kv_set("mental", _dump(m))

    def counter(self, k: str, inc: int = 0) -> int:
        v = int(self._kv_get(f"ctr:{k}") or 0) + inc
        if inc:
            self._kv_set(f"ctr:{k}", str(v))
        return v

    # ---- evidence classification cache: each (claim, episode) pair is judged once ----
    def get_classification(self, claim_id: str, episode_id: str):
        r = self.conn.execute("SELECT polarity, note FROM evidence_class WHERE claim_id=? AND episode_id=?",
                              (claim_id, episode_id)).fetchone()
        return (r["polarity"], r["note"]) if r else None

    def set_classification(self, claim_id: str, episode_id: str, polarity: str, note: str) -> None:
        self.conn.execute("INSERT OR REPLACE INTO evidence_class VALUES (?,?,?,?)", (claim_id, episode_id, polarity, note))
        self.conn.commit()

    # ---- snapshots (for clone experiments) ----
    def snapshot(self, name: str) -> dict:
        doc = {
            "affect": self.load_affect().model_dump(mode="json"),
            "mental": self.load_mental().model_dump(mode="json"),
            "claims": [c.model_dump(mode="json") for c in self.claims(include_inactive=True)],
            "questions": [q.model_dump(mode="json") for q in self.questions(status=None)],
            "episode_count": self.episode_count(),
        }
        self.conn.execute("INSERT OR REPLACE INTO snapshots VALUES (?,?,?)",
                          (name, now().isoformat(), json.dumps(doc)))
        self.conn.commit()
        self.audit("snapshot", name=name)
        return doc

    def snapshots(self) -> list[str]:
        return [r["name"] for r in self.conn.execute("SELECT name FROM snapshots ORDER BY ts")]

    def clone_to(self, dest: Path) -> None:
        """Byte-level copy of the whole home DB (for A/B clone experiments)."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(dest)) as out:
            self.conn.backup(out)
