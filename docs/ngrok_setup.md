# Running YorubaBridgeAI + ngrok after a Mac reboot

Quick-restart commands. Run each in Terminal; ngrok reuses its saved config.

## 1. Start supporting services (if not already running)

```bash
cd ~/Desktop/WORK/AI-projects/YorubaBridgeAI
docker compose up -d postgres redis     # skip if they're still up
```

## 2. Start the API + web UI on port 8081

Port 8080 is taken by another local service (Docker) — YorubaBridgeAI uses
**8081**. Run inside the `tmux`/`nohup` form so it survives terminal close:

```bash
cd ~/Desktop/WORK/AI-projects/YorubaBridgeAI
nohup .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8081 \
  > /tmp/yb_8081.log 2>&1 &
```

Model load takes ~10–20 s (5.2 GB from local disk). Verify:

```bash
curl http://127.0.0.1:8081/health
# {"status":"ok","app":"YorubaBridgeAI",...}
```

## 3. Start the ngrok tunnel

```bash
nohup ngrok http 8081 --log stdout --log-format logfmt > /tmp/ngrok_yb.log 2>&1 &
```

Find the public URL (it may change after reboots on the free plan):

```bash
curl -s http://127.0.0.1:4040/api/tunnels | python3 -c \
  "import json,sys; [print(t['public_url']) for t in json.load(sys.stdin)['tunnels']]"
```

## 4. WhatsApp webhook URL

Append the webhook path to whatever URL ngrok gives:

```
<public-url>/api/v1/webhook
```

Example with today's tunnel:
`https://cylinder-scored-conch.ngrok-free.dev/api/v1/webhook`

Register that URL in the Meta App Dashboard (WhatsApp → Configuration →
Webhook → Callback URL) with the verify token from `.env`
(`WHATSAPP_VERIFY_TOKEN`).

## 5. Stop everything (when needed)

```bash
pkill -f "uvicorn app.main:app --host 127.0.0.1 --port 8081"
pkill -f "ngrok http 8081"
```

## Notes

- The free ngrok plan issues a **new random URL per tunnel session** —
  update the webhook URL in the Meta dashboard after each reboot, or wire
  the tunnel API call into a small script.
- The 5.2 GB model stays on disk; nothing is re-downloaded.
- `.env` holds all credentials and is gitignored.
