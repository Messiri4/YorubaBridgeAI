# Infrastructure report — Docker, PostgreSQL, Redis

Date: 2026-11-06 · Branch: `infrastructure-review` · Machine: Intel Mac (x64, macOS)

## 1. What was wrong with Docker

Docker Desktop's daemon was unresponsive: `docker` CLI commands hung until the
120 s timeout, and even a direct HTTP `/_ping` on the daemon socket never
returned.

Diagnosis from logs and disk state:

- **Host disk exhaustion (primary cause).** The data volume
  (`/System/Volumes/Data`) had only **1.6 GiB free** (100% capacity) at the
  start of this work. The Docker Desktop VM disk
  (`~/Library/Containers/com.docker.docker/Data/vms/0/data/Docker.raw`) holds
  ~28 GiB of images/build cache on top of that.
- **VM-side `input/output error`s.** `dockerd.log` (VM) and
  `com.docker.backend.log` (host) were full of
  `write /var/lib/desktop-containerd/.../meta.db: input/output error` —
  containerd could not write image blobs or its bolt metadata DB. Every image
  resolve, container create, and build failed; the daemon stopped answering.
- Docker Desktop had been running for 3+ days with the backend wedged in this
  state (backend PID alive since Nov 2, no log output since Nov 4 18:38).
- Docker CLI itself was fine (27.5.1) and the socket
  (`~/.docker/run/docker.sock`) existed; the problem was entirely inside the VM.

## 2. What was done to fix it (non-destructive)

1. Freed host disk space by deleting only **regenerable caches** (no user data,
   no Docker images/volumes touched):
   - pip caches (`~/.cache/pip`, `~/Library/Caches/pip`)
   - npm cacache, Yarn cache
   - Spotify / WhatsApp / Telegram / Unity Hub app caches
   - Chrome cache subfolders (Default/Cache, Code Cache, GPUCache)
   - Hugging Face `xet` cache (the NLLB model blob in
     `~/.cache/huggingface/hub` was **kept** — the app needs it offline)
   - Result: 1.6 GiB → ~6 GiB free.
2. Quit Docker Desktop via AppleScript, force-killed the wedged
   `com.docker.backend` process (the only "force" action; nothing destructive
   to containers/images/volumes), and relaunched Docker Desktop.
3. After restart: `/_ping` → `OK`, `docker ps` responsive, existing images and
   containers intact. Notably the other project's `airbnb_predictor` containers
   auto-restarted (`restart: unless-stopped`) — evidence Docker is fully
   functional again.

Remaining risk: `docker builder prune` and even `docker system df` still
stall, so some containerd metadata damage likely persists inside the VM. With
~6 GiB free the daemon works, but if errors recur the recommended (manual,
destructive — **needs your approval**) fix is freeing more disk (the disk is
100% full again: 4.1 GiB free) and, only if I/O errors return, using Docker
Desktop → Troubleshoot → "Clean / Purge data", which resets images/volumes.

**Note:** no `docker system prune` was run — it touches images/containers and
was out of scope without your approval.

## 3. PostgreSQL — functioning ✅

- Started via `docker compose up -d postgres` (`postgres:16-alpine`),
  port 5432 reachable on localhost, `pg_isready` → accepting connections.
- Verified with the project's real `app/services/history.py` `HistoryStore`:
  - `record()` inserted a real translation row (Yoruba diacritics intact:
    `Ẹ kú àárọ̀, báwo ni?`), auto-created `translation_history` schema.
  - `recent()` returned the row correctly (id, timestamps, latency, flags).
  - Error handling: a store built on a failing connection returns `False`/`[]`
    without raising (degrade-safe design confirmed on real Postgres).
  - Recovery: after the simulated failure, the real store inserted again
    successfully (2 rows total). No existing user data deleted (the table was
    created fresh; only test rows from this session exist).

## 4. Redis caching — functioning ✅

- Started via `docker compose up -d redis` (`redis:7-alpine`), port 6379,
  `PING` → `PONG`.
- Tested with the real `TranslationCache` + real Redis (no mocks):
  - First request: cache **miss** (`None`).
  - `set()` then `get()`: cache **hit**, payload matches the original exactly.
  - TTL: key created with the configured `cache_ttl_seconds` (3600 s); a key
    with `ex=2` disappeared after 3 s → expiration works.
  - Rate limiting (`RateLimiter` with real Redis): 10 requests allowed, the
    11th and 12th got HTTP 429, matching `rate_limit_max_requests=10`.
  - Outage handling: a cache pointing at an unreachable Redis (port 9999)
    disables itself at construction; `get()` → `None`, `set()` no-op, no
    exceptions — translation would continue uncached.

## 5. Integration tests

- Full suite: `pytest tests` → **65 passed, 1 skipped, 3 deselected** (11.4 s).
  These are the project's existing tests (unit-level, mocked Redis/DB where the
  tests mock).
- **Real-service verification (not mocked)**, run live against the Docker
  Postgres and Redis:
  - HistoryStore insert/retrieve/error/recovery — real Postgres (see §3).
  - Cache miss/hit/TTL/expiration/outage + rate limit 429 — real Redis
    (see §4).
  - FastAPI app boot with real services: `create_app()` logged
    `Redis connected for caching and rate limiting`, and `GET /health`
    returned `200 {"status": "ok", ...}` via TestClient.
- Credentials: read from `.env` via pydantic-settings (`DATABASE_URL`,
  `REDIS_URL`, `DATABASE_ENABLED=true`); `.env` is git-ignored and was created
  from `.env.example`.

## 6. Remaining issues

- Host disk is again at 100% (4.1 GiB free). Recommend cleaning more space;
  Docker Desktop misbehaves below a few GiB.
- Docker VM containerd metadata may still be corrupted (build/prune commands
  stall). If builds fail with `input/output error` again, a Docker Desktop
  "Clean / Purge data" (destructive — ask first) or a Docker.raw re-create is
  the fix.
- Compose hardcodes `POSTGRES_PASSWORD: yuba_change_me` in
  `docker-compose.yml`; move to `${POSTGRES_PASSWORD}` from `.env` when
  convenient.
- NLLB model (~4.6 GB) lives in `~/.cache/huggingface/hub` — kept intact for
  offline translation use.

## 7. How to start and stop services yourself

```bash
cd /Users/user/Desktop/WORK/AI-projects/YorubaBridgeAI-infra

# Start Docker Desktop if not running
open -a Docker   # wait until `docker info` responds

# Start the app's services (Postgres 5432, Redis 6379)
docker compose up -d postgres redis

# Stop only these services
docker compose stop postgres redis

# Stop and remove the containers (the pgdata volume is kept)
docker compose down

# Everything including the database volume (DELETES stored history)
docker compose down -v

# Quit Docker Desktop entirely
osascript -e 'quit app "Docker"'
```

`.env` needs: `DATABASE_URL=postgresql://yoruba:<password>@localhost:5432/yorubabridge`,
`REDIS_URL=redis://localhost:6379/0`, `DATABASE_ENABLED=true`.
