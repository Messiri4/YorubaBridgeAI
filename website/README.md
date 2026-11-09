# YorubaBridgeAI Public Website

Static, dependency-free public information site. It is **completely separate**
from the FastAPI application: it is never served by the API and shares no
code or build step. Deploy it independently to Cloudflare Pages.

## Pages

| URL        | File                   |
|------------|------------------------|
| `/`        | `index.html`           |
| `/privacy` | `privacy/index.html`   |
| `/terms`   | `terms/index.html`     |
| `/contact` | `contact/index.html`   |

Directory-style pages mean `/privacy`, `/terms`, and `/contact` are real
files on disk — direct navigation and refresh work out of the box on
Cloudflare Pages. No redirects, rewrites, or SPA routing are required.

No analytics, cookies, forms, or third-party scripts are included.

Contact: messirimiracle10@gmail.com

## Deploy to Cloudflare Pages

1. Connect the Git repository to a new Cloudflare Pages project.
2. Build settings:
   - **Build command:** none (leave empty)
   - **Build output directory:** `website`
3. Deploy. Pages serves every file under `website/` at the site root, so
   `/`, `/privacy`, `/terms`, and `/contact` (with `index.html` resolution)
   all work over HTTPS, including direct navigation and refresh.
4. Optional: attach a custom domain in Pages settings.

## Before going live (launch blockers)

- **[BLOCKER] Effective date placeholders** — `privacy/index.html` and
  `terms/index.html` both contain a `data-launch-blocker="effective-date"`
  span reading "TO BE SET ON PUBLICATION". Replace both with the actual
  publication date before publishing.
- Confirm the final wording of the privacy policy and terms.
- The Meta app review configuration should reference the published
  `https://<your-domain>/privacy` URL.
- The contact email is publicly visible and may receive spam.

## Run locally

```bash
cd website
python3 -m http.server 8090
# visit http://localhost:8090/privacy etc.
```

Tests live in `tests/test_website.py` and run with the normal suite
(`pytest tests/test_website.py`). They validate internal links, the
stylesheet reference, the absence of analytics/tracking, and the presence
of the contact email.
