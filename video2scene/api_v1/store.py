import json
import copy
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from video2scene.contracts_v1 import CONTRACT_VERSION, ContractError, canonical_hash, utc_now, validate


class HostStore:
    def __init__(self, root):
        root = Path(root); root.mkdir(parents=True, exist_ok=True)
        self.path = root / "host.sqlite"
        with self.connect() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS commands(scene TEXT, id TEXT, hash TEXT, command TEXT, record TEXT, PRIMARY KEY(scene,id));
            CREATE TABLE IF NOT EXISTS requests(scene TEXT, id TEXT, hash TEXT, request TEXT, response TEXT, PRIMARY KEY(scene,id));
            CREATE TABLE IF NOT EXISTS events(scene TEXT, seq INTEGER, id TEXT UNIQUE, body TEXT, PRIMARY KEY(scene,seq));
            CREATE TABLE IF NOT EXISTS artifacts(uri TEXT PRIMARY KEY, ref TEXT);
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.execute("PRAGMA journal_mode=WAL"); db.execute("PRAGMA synchronous=FULL")
        try:
            with db:
                yield db
        finally:
            db.close()

    def accept_command(self, command):
        validate(command, "UserCommand")
        scene, key, fingerprint = command["scene_id"], command["command_id"], canonical_hash(command)
        receipt = {"command_id": key, "operation_id": "op_" + uuid.uuid4().hex, "state": "accepted"}
        record = {"receipt": receipt, "state": "accepted", "request_ids": [], "message": "", "error": None}
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT hash,record FROM commands WHERE scene=? AND id=?", (scene,key)).fetchone()
            if old:
                if old[0] != fingerprint: raise ContractError("IDEMPOTENCY_CONFLICT", "Command ID already has different content")
                return json.loads(old[1]), False
            db.execute("INSERT INTO commands VALUES(?,?,?,?,?)", (scene,key,fingerprint,json.dumps(command),json.dumps(record)))
        return record, True

    def command(self, scene, key):
        with self.connect() as db:
            row = db.execute("SELECT record FROM commands WHERE scene=? AND id=?", (scene,key)).fetchone()
        if not row: raise ContractError("NOT_FOUND", "Command was not found")
        return json.loads(row[0])

    def save_command(self, scene, key, record):
        validate(record,"CommandRecord")
        with self.connect() as db: db.execute("UPDATE commands SET record=? WHERE scene=? AND id=?", (json.dumps(record),scene,key))

    def pending_commands(self):
        with self.connect() as db: rows=db.execute("SELECT command,record FROM commands").fetchall()
        return [json.loads(c) for c,r in rows if json.loads(r)["state"] in ("accepted","running")]

    def request(self, scene, key):
        with self.connect() as db:
            row=db.execute("SELECT response FROM requests WHERE scene=? AND id=?",(scene,key)).fetchone()
        if not row or row[0] is None: raise ContractError("NOT_FOUND", "Request result unavailable; execution may still be pending")
        return json.loads(row[0])

    def reserve_request(self, request):
        scene,key=request["scene_id"],request["request_id"]; fingerprint=canonical_hash(request)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            old=db.execute("SELECT hash,response FROM requests WHERE scene=? AND id=?",(scene,key)).fetchone()
            if old:
                if old[0]!=fingerprint: raise ContractError("IDEMPOTENCY_CONFLICT", "Request ID has different content")
                return json.loads(old[1]) if old[1] else None, False
            db.execute("INSERT INTO requests VALUES(?,?,?,?,NULL)",(scene,key,fingerprint,json.dumps(request)))
        return None, True

    def save_request(self, request, response):
        self.register_artifacts(response)
        with self.connect() as db: db.execute("UPDATE requests SET response=? WHERE scene=? AND id=?",(json.dumps(response),request["scene_id"],request["request_id"]))

    def register_artifacts(self, value):
        if isinstance(value,dict):
            if set(value)=={"uri","sha256","media_type","role"}:
                validate(value,"ArtifactRef")
                with self.connect() as db:
                    old=db.execute("SELECT ref FROM artifacts WHERE uri=?",(value["uri"],)).fetchone()
                    if old and json.loads(old[0])["sha256"]!=value["sha256"]:
                        raise ContractError("STALE_ARTIFACT","Artifact URI cannot acquire different content")
                    db.execute("INSERT OR IGNORE INTO artifacts VALUES(?,?)",(value["uri"],json.dumps(value)))
            else:
                for child in value.values():self.register_artifacts(child)
        elif isinstance(value,list):
            for child in value:self.register_artifacts(child)

    def artifact(self, uri):
        with self.connect() as db:row=db.execute("SELECT ref FROM artifacts WHERE uri=?",(uri,)).fetchone()
        if not row:raise ContractError("NOT_FOUND","Artifact URI has not been published by a tool")
        return json.loads(row[0])

    def emit(self, scene, kind, data, request_id=None, event_id=None):
        data=copy.deepcopy(data)
        event_id=event_id or "evt_"+uuid.uuid4().hex
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            old=db.execute("SELECT body FROM events WHERE id=?",(event_id,)).fetchone()
            if old: return json.loads(old[0])
            if kind=="job.updated":
                rank={"queued":0,"running":1,"cancel_requested":2,"succeeded":3,"failed":3,"cancelled":3,"stale":3}
                for row in db.execute("SELECT body FROM events WHERE scene=? ORDER BY seq DESC",(scene,)):
                    previous=json.loads(row[0])
                    if previous["type"]==kind and previous["data"]["job_id"]==data["job_id"]:
                        before=previous["data"]["state"]
                        if rank[before]==3 or rank[data["state"]]<rank[before]:return previous
                        break
            seq=db.execute("SELECT COALESCE(MAX(seq),0)+1 FROM events WHERE scene=?",(scene,)).fetchone()[0]
            event={"contract_version":CONTRACT_VERSION,"event_id":event_id,"scene_id":scene,"seq":seq,"type":kind,"occurred_at":utc_now(),"request_id":request_id,"data":data}
            validate(event,"SceneEvent")
            db.execute("INSERT INTO events VALUES(?,?,?,?)",(scene,seq,event_id,json.dumps(event)))
        return event

    def events(self, scene, after=0):
        if after<0: raise ContractError("INVALID_ARGUMENT","Event cursor must be nonnegative")
        with self.connect() as db: rows=db.execute("SELECT body FROM events WHERE scene=? AND seq>? ORDER BY seq",(scene,after)).fetchall()
        return [json.loads(row[0]) for row in rows]

