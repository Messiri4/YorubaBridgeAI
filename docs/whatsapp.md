# WhatsApp Cloud API Integration

Architecture, local testing, and deployment guide for the WhatsApp
integration.

## Architecture

```
Meta servers
   │  (HTTPS, X-Hub-Signature-256)
   ▼
GET/POST /api/v1/webhook        app/api/v1/routes_whatsapp.py
   │
   ├─ GET: verification handshake (hub.verify_token vs WHATSAPP_VERIFY_TOKEN)
   ├─ POST: verify_signature()      app/services/whatsapp_webhook.py
   │         HMAC-SHA256 over the RAW request body (never re-serialized JSON)
   ├─ extract_text_messages()       same file — entries→changes→values→messages
   │         text messages only; image/audio/sticker types ignored gracefully
   ├─ MessageDeduplicator.seen()    app/services/whatsapp_dedup.py
   │         SET NX wamid key in Redis (7-day TTL); Meta retries must not
   │         produce duplicate replies. In-memory fallback when Redis is down.
   ├─ RateLimiter.check("wa:<sender>")   app/core/rate_limit.py
   ├─ guess_language()              yor/eng heuristic → translation direction
   ├─ TranslationService.translate()   app/services/translation.py
   │         (reuses NLLB-600M + phrase overrides + cache — no logic duplicated)
   └─ WhatsAppClient.send_text()    app/services/whatsapp_client.py
             POST graph.facebook.com/v20.0/<PHONE_NUMBER_ID>/messages
             Bearer token from env only
```

Privacy: message bodies and phone numbers are never logged — only counts
(`received/deduped/replied/failed`).

## Environment variables (all from Meta App Dashboard)

| Variable                   | Meaning                                                           |
| -------------------------- | ----------------------------------------------------------------- |
| `WHATSAPP_TOKEN`           | permanent access token with `whatsapp_business_messaging`         |
| `WHATSAPP_PHONE_NUMBER_ID` | the business phone number's ID                                    |
| `WHATSAPP_VERIFY_TOKEN`    | any random string YOU choose; must match Meta's subscription form |
| `WHATSAPP_APP_SECRET`      | App Settings → Basic → App secret (used for HMAC)                 |

## Local testing without credentials

The whole pipeline is testable with no Meta account:

```bash
pytest tests/test_whatsapp.py tests/test_whatsapp_dedup.py -q
```

Live-Redis demo (signature + dedup against real Redis, Graph API mocked):

```bash
.venv/bin/python - <<'EOF'
import json, os, hmac, hashlib
os.environ['WHATSAPP_APP_SECRET'] = 'local-test-secret-ONLY'
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from app.main import create_app

payload = {'entry':[{'changes':[{'value':{'messages':[
  {'id':'wamid.DEMO1','from':'234000000000','type':'text','text':{'body':'Ẹ kú àárọ̀, ọrẹ́ mi'}}]}}]}]}
body = json.dumps(payload).encode()
sig = 'sha256=' + hmac.new(b'local-test-secret-ONLY', body, hashlib.sha256).hexdigest()

app = create_app(load_model=False)
# inject a fake translation service (or run with the real model)
from unittest.mock import MagicMock
svc = MagicMock()
svc.translate.return_value.__dict__ = {'translated_text':'Good morning, my friend.',
                                       'source_language':'yor','target_language':'eng','latency_ms':0.5}
app.state.translation_service = svc
with TestClient(app) as c:
    with patch('app.api.v1.routes_whatsapp.WhatsAppClient') as W:
        inst = W.return_value; inst.send_text.return_value = {'messages':[{'id':'wamid.out'}]}
        print(c.post('/api/v1/webhook', content=body, headers={'X-Hub-Signature-256': sig}).json())
        print(c.post('/api/v1/webhook', content=body, headers={'X-Hub-Signature-256': sig}).json())  # deduped
EOF
```

## Connecting a real test number (deployment steps)

1. Create a Meta developer account → create a Business app with the
   _WhatsApp_ product (free; the test number sends to up to 5 verified
   recipients at no charge).
2. Copy the four variables above into `.env`.
3. Expose the webhook publicly — locally use a tunnel:
   `ngrok http 8000` → note the HTTPS URL.
4. In App Dashboard → WhatsApp → Configuration → Webhook:
   - Callback URL: `<tunnel>/api/v1/webhook`
   - Verify token: the value of `WHATSAPP_VERIFY_TOKEN` → Meta performs the
     GET handshake automatically.
5. Subscribe to the `messages` webhook field.
6. Send a WhatsApp message from the verified recipient phone to the test
   number; watch `logs` for `Webhook batch: 1 received, 0 deduped, 1 replied`.

## Remaining before production

- Real Meta credentials + public HTTPS URL (steps above) — the code is ready;
  only account setup remains.
- Media/voice note ingestion through WhatsApp (the `/voice-translate` pipeline
  exists; the webhook currently handles `text` messages only).
- Interactive messages (buttons/lists), delivery/read receipts, and
  long-lived-token refresh automation.
