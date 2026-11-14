#!/usr/bin/env python3
"""Tiny receiver for the NarutoCV collection tool.

Deps (not part of the project's pyproject on purpose):  pip install aiohttp jsonschema

Run:
    COLLECT_TOKEN=some-shared-secret python receive.py --out data/raw/collection --port 8787
Self-test (no network, temp dir):
    python receive.py --self-test
Delete one subject (honouring a withdrawal request):
    python receive.py --out data/raw/collection --delete-subject S-7K3QMD

API (all but /health need header X-Collect-Token):
    PUT /api/v1/{subject_id}/{session_id}/clips/{clip_id}.{webm|mp4}   body = video bytes
    PUT /api/v1/{subject_id}/{session_id}/session.json                 body = session.json (schema-validated)
    GET /api/v1/health
Layout written:  <out>/<subject_id>/<session_id>/{session.json, clips/<clip_id>.<ext>}
The client uploads clips first and session.json last, so session.json marks a finished upload.
"""

from __future__ import annotations

import argparse
import asyncio
import hmac
import json
import os
import re
import shutil
import ssl
import sys
import tempfile
from pathlib import Path

from aiohttp import web
from jsonschema import Draft202012Validator

ID_RE = re.compile(r"^[A-Za-z0-9_-]{4,64}$")
CLIP_RE = re.compile(r"^([A-Za-z0-9_-]{1,64})\.(webm|mp4)$")
DEFAULT_SCHEMA = (
    Path(__file__).resolve().parents[3] / "schemas" / "collection_session_v1.schema.json"
)
CHUNK = 1 << 20


def err(status: int, msg: str) -> web.Response:
    return web.json_response({"error": msg}, status=status)


@web.middleware
async def cors_and_auth(request: web.Request, handler):
    cfg = request.app["cfg"]
    cors = {
        "Access-Control-Allow-Origin": cfg["allow_origin"],
        "Access-Control-Allow-Methods": "PUT, GET, OPTIONS",
        "Access-Control-Allow-Headers": "X-Collect-Token, Content-Type",
        "Access-Control-Max-Age": "600",
    }
    if request.method == "OPTIONS":
        return web.Response(status=204, headers=cors)
    if request.path != "/api/v1/health":
        sent = request.headers.get("X-Collect-Token", "")
        if not hmac.compare_digest(sent.encode(), cfg["token"].encode()):
            return web.json_response({"error": "bad or missing token"}, status=401, headers=cors)
    try:
        resp = await handler(request)
    except web.HTTPException as e:  # includes 404/405/413
        resp = web.json_response({"error": e.reason}, status=e.status)
    resp.headers.update(cors)
    return resp


async def read_capped(request: web.Request, limit: int) -> bytes | None:
    """Read the whole body, or None if it exceeds `limit`."""
    declared = request.content_length
    if declared is not None and declared > limit:
        return None
    buf = bytearray()
    async for chunk in request.content.iter_chunked(CHUNK):
        buf += chunk
        if len(buf) > limit:
            return None
    return bytes(buf)


def _target_dir(request: web.Request) -> tuple[Path | None, web.Response | None]:
    subject = request.match_info["subject_id"]
    session = request.match_info["session_id"]
    if not (ID_RE.match(subject) and ID_RE.match(session)):
        return None, err(400, "invalid subject_id/session_id")
    base: Path = request.app["cfg"]["out"]
    d = (base / subject / session).resolve()
    if base.resolve() not in d.parents:
        return None, err(400, "bad path")
    return d, None


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(data)
    os.replace(tmp, path)


async def put_clip(request: web.Request) -> web.Response:
    d, bad = _target_dir(request)
    if bad:
        return bad
    m = CLIP_RE.match(request.match_info["filename"])
    if not m:
        return err(400, "clip filename must be <clip_id>.webm|mp4")
    body = await read_capped(request, request.app["cfg"]["max_clip_bytes"])
    if body is None:
        return err(413, "clip too large")
    if not body:
        return err(400, "empty body")
    _atomic_write(d / "clips" / m.group(0), body)
    return web.json_response({"ok": True, "bytes": len(body)})


async def put_session(request: web.Request) -> web.Response:
    d, bad = _target_dir(request)
    if bad:
        return bad
    body = await read_capped(request, 2 * 1024 * 1024)
    if body is None:
        return err(413, "session.json too large")
    try:
        doc = json.loads(body)
    except ValueError:
        return err(400, "session.json is not valid JSON")
    errors = sorted(request.app["validator"].iter_errors(doc), key=lambda e: list(e.path))
    if errors:
        e = errors[0]
        return err(422, f"schema violation at /{'/'.join(map(str, e.path))}: {e.message}"[:500])
    if (
        doc["subject"]["subject_id"] != request.match_info["subject_id"]
        or doc["session"]["session_id"] != request.match_info["session_id"]
    ):
        return err(400, "ids in session.json do not match the URL")
    missing = [c["file"] for c in doc["clips"] if not (d / c["file"]).is_file()]
    _atomic_write(d / "session.json", json.dumps(doc, indent=2).encode())
    # 200 even with missing clips: client can see what to resend; session.json is kept either way.
    return web.json_response(
        {"ok": True, "clips_declared": len(doc["clips"]), "missing_files": missing}
    )


async def health(_: web.Request) -> web.Response:
    return web.json_response({"ok": True})


def make_app(
    out: Path,
    token: str,
    schema_path: Path = DEFAULT_SCHEMA,
    max_clip_mb: int = 200,
    allow_origin: str = "*",
) -> web.Application:
    schema = json.loads(schema_path.read_text())
    Draft202012Validator.check_schema(schema)
    app = web.Application(middlewares=[cors_and_auth])
    app["cfg"] = {
        "out": out,
        "token": token,
        "max_clip_bytes": max_clip_mb * 1024 * 1024,
        "allow_origin": allow_origin,
    }
    app["validator"] = Draft202012Validator(schema)
    app.router.add_get("/api/v1/health", health)
    app.router.add_put("/api/v1/{subject_id}/{session_id}/clips/{filename}", put_clip)
    app.router.add_put("/api/v1/{subject_id}/{session_id}/session.json", put_session)
    return app


def delete_subject(out: Path, subject_id: str) -> bool:
    if not ID_RE.match(subject_id):
        raise SystemExit("invalid subject id")
    d = out / subject_id
    if d.is_dir():
        shutil.rmtree(d)
        return True
    return False


# ---------------------------------------------------------------------------- self test
def _sample_session(subject: str, session: str, clip_id: str) -> dict:
    return {
        "schema_version": "collection_session_v1",
        "subject": {
            "subject_id": subject,
            "camera_type": "laptop",
            "lighting": "normal",
            "background": "plain",
            "distance_m": 1,
            "dominant_hand": "R",
            "monk_skin_tone": None,
            "consent": {
                "accepted": True,
                "accepted_at": "2026-11-13T10:00:00.000Z",
                "text_hash": "deadbeef",
            },
        },
        "session": {
            "session_id": session,
            "started_at": "2026-11-13T10:00:00.000Z",
            "ended_at": None,
            "status": "in_progress",
        },
        "device": {
            "user_agent": "selftest",
            "platform": "x",
            "screen": {"width": 1, "height": 1, "device_pixel_ratio": 1},
            "requested": {"width": 1280, "height": 720, "fps": 30},
            "achieved": {"width": 1280, "height": 720, "fps_measured": 30},
            "mirrored_display": True,
        },
        "protocol": {"version": "1.0.0", "seed": "S", "config": {}},
        "clips": [
            {
                "clip_id": clip_id,
                "block": "seal",
                "label": "rat",
                "take": 1,
                "attempt": 1,
                "mime": "video/webm",
                "file": f"clips/{clip_id}.webm",
                "size_bytes": 5,
                "duration_ms": 9800,
                "width": 1280,
                "height": 720,
                "fps_measured": 30.0,
                "phase_offsets_ms": {
                    "form_start": 3000,
                    "hold_start": 5000,
                    "hold_end": 8000,
                    "release_end": 9500,
                },
            }
        ],
    }


async def self_test() -> None:
    from aiohttp.test_utils import TestClient, TestServer

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "collection"
        app = make_app(out, token="secret", max_clip_mb=1)
        async with TestClient(TestServer(app)) as c:
            H = {"X-Collect-Token": "secret"}
            base = "/api/v1/S-TEST01/sess-0001"
            checks: list[tuple[str, bool]] = []

            def check(name: str, cond: bool) -> None:
                checks.append((name, cond))
                print(("PASS " if cond else "FAIL ") + name)

            check("health needs no token", (await c.get("/api/v1/health")).status == 200)
            check(
                "missing token -> 401",
                (await c.put(f"{base}/clips/c001.webm", data=b"x")).status == 401,
            )
            check(
                "wrong token -> 401",
                (
                    await c.put(
                        f"{base}/clips/c001.webm", data=b"x", headers={"X-Collect-Token": "nope"}
                    )
                ).status
                == 401,
            )
            r = await c.options(f"{base}/session.json")
            check(
                "CORS preflight 204 without token",
                r.status == 204 and "X-Collect-Token" in r.headers["Access-Control-Allow-Headers"],
            )
            r = await c.put(f"{base}/clips/c001.webm", data=b"hello", headers=H)
            check(
                "clip stored",
                r.status == 200
                and (out / "S-TEST01/sess-0001/clips/c001.webm").read_bytes() == b"hello",
            )
            check("no .part files left", not list(out.rglob("*.part")))
            r = await c.put(f"{base}/clips/c002.webm", data=b"x" * (1024 * 1024 + 1), headers=H)
            check(
                "oversize clip -> 413",
                r.status == 413 and not (out / "S-TEST01/sess-0001/clips/c002.webm").exists(),
            )
            check(
                "bad extension -> 400",
                (await c.put(f"{base}/clips/c003.exe", data=b"x", headers=H)).status == 400,
            )
            check(
                "bad clip name -> 400",
                (await c.put(f"{base}/clips/..%2Fevil.webm", data=b"x", headers=H)).status
                in (400, 404),
            )
            check(
                "bad subject id -> 400",
                (await c.put("/api/v1/a/sess-0001/clips/c1.webm", data=b"x", headers=H)).status
                == 400,
            )
            check(
                "path traversal -> rejected",
                (
                    await c.put("/api/v1/..%2F..%2Fx/sess-0001/clips/c1.webm", data=b"x", headers=H)
                ).status
                in (400, 404),
            )

            good = _sample_session("S-TEST01", "sess-0001", "c001")
            r = await c.put(f"{base}/session.json", data=json.dumps(good), headers=H)
            body = await r.json()
            check("valid session.json accepted", r.status == 200 and body["missing_files"] == [])
            check("session.json on disk", (out / "S-TEST01/sess-0001/session.json").is_file())

            bad = json.loads(json.dumps(good))
            bad["subject"]["monk_skin_tone"] = 99
            check(
                "schema violation -> 422",
                (await c.put(f"{base}/session.json", data=json.dumps(bad), headers=H)).status
                == 422,
            )
            bad = json.loads(json.dumps(good))
            bad["subject"]["email"] = "a@b.c"
            check(
                "extra PII field -> 422",
                (await c.put(f"{base}/session.json", data=json.dumps(bad), headers=H)).status
                == 422,
            )
            check(
                "non-JSON -> 400",
                (await c.put(f"{base}/session.json", data=b"{nope", headers=H)).status == 400,
            )
            mismatch = _sample_session("S-OTHER1", "sess-0001", "c001")
            check(
                "id mismatch -> 400",
                (await c.put(f"{base}/session.json", data=json.dumps(mismatch), headers=H)).status
                == 400,
            )

        check(
            "delete-subject removes the folder",
            delete_subject(out, "S-TEST01") and not (out / "S-TEST01").exists(),
        )
        failed = [n for n, ok in checks if not ok]
        print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed")
        if failed:
            raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, default=Path("data/raw/collection"))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument(
        "--token",
        default=os.environ.get("COLLECT_TOKEN"),
        help="shared secret (or env COLLECT_TOKEN)",
    )
    ap.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    ap.add_argument("--max-clip-mb", type=int, default=200)
    ap.add_argument("--allow-origin", default="*", help="CORS origin, e.g. https://you.github.io")
    ap.add_argument("--cert", type=Path, help="TLS cert (PEM) to serve https directly")
    ap.add_argument("--key", type=Path, help="TLS key (PEM)")
    ap.add_argument(
        "--delete-subject",
        metavar="SUBJECT_ID",
        help="delete everything stored for this subject and exit",
    )
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        asyncio.run(self_test())
        return
    if a.delete_subject:
        print(
            "deleted"
            if delete_subject(a.out, a.delete_subject)
            else "nothing found for that subject"
        )
        return
    if not a.token:
        sys.exit("refusing to start without a token: pass --token or set COLLECT_TOKEN")
    a.out.mkdir(parents=True, exist_ok=True)
    ctx = None
    if a.cert and a.key:
        ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ctx.load_cert_chain(a.cert, a.key)
    app = make_app(a.out, a.token, a.schema, a.max_clip_mb, a.allow_origin)
    web.run_app(
        app,
        host=a.host,
        port=a.port,
        ssl_context=ctx,
        client_max_size=a.max_clip_mb * 1024 * 1024 + 4096,
    )


if __name__ == "__main__":
    main()
