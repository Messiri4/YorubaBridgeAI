"""Tests for the static public website under website/.

The site is fully static and deployed separately from the FastAPI app
(see website/README.md). These tests guard its integrity: internal links
must resolve, no analytics/tracking may appear, and the contact email must
be present on every page.
"""

from __future__ import annotations

import re
from pathlib import Path

WEBSITE = Path(__file__).resolve().parent.parent / "website"

PAGES: dict[str, Path] = {
    "/": WEBSITE / "index.html",
    "/privacy": WEBSITE / "privacy" / "index.html",
    "/terms": WEBSITE / "terms" / "index.html",
    "/contact": WEBSITE / "contact" / "index.html",
}

CONTACT_EMAIL = "messirimiracle10@gmail.com"


def test_all_pages_exist() -> None:
    for url, path in PAGES.items():
        assert path.is_file(), f"missing page for {url}: {path}"


def test_internal_links_resolve() -> None:
    """Every internal href on every page maps to a real file."""
    resolver: dict[str, Path] = {
        "/": WEBSITE / "index.html",
        "/privacy": WEBSITE / "privacy" / "index.html",
        "/terms": WEBSITE / "terms" / "index.html",
        "/contact": WEBSITE / "contact" / "index.html",
    }
    for url, path in PAGES.items():
        html = path.read_text(encoding="utf-8")
        hrefs = set(re.findall(r'href="([^"]+)"', html))
        for href in hrefs:
            if href.startswith(("mailto:", "http://", "https://")):
                continue
            if href.endswith(".css"):  # stylesheet, checked separately
                continue
            assert href in resolver, f"{url} links to unknown internal path {href}"
            assert resolver[href].is_file(), f"{url} -> {href} target missing"


def test_stylesheet_reference_consistent() -> None:
    """All pages must reference the same absolute stylesheet /styles.css."""
    for url, path in PAGES.items():
        html = path.read_text(encoding="utf-8")
        styles = re.findall(r'href="([^"]*styles\.css)"', html)
        assert styles, f"{url} does not reference styles.css"
        assert all(s == "/styles.css" for s in styles), (
            f"{url} uses inconsistent stylesheet reference: {styles}"
        )
    assert (WEBSITE / "styles.css").is_file()


def test_no_analytics_or_tracking() -> None:
    """No external scripts, analytics, or tracking pixels anywhere."""
    forbidden = re.compile(
        r"(google-analytics|googletagmanager|gtag\(|fbq\(|hotjar|mixpanel"
        r"|segment\.com|plausible|matomo|fathom|clarity\.ms|<script)",
        re.IGNORECASE,
    )
    for url, path in PAGES.items():
        html = path.read_text(encoding="utf-8")
        assert not forbidden.search(html), f"{url} contains analytics/script tags"
        # no external resource loads at all (fonts, CDNs, images)
        externals = re.findall(r'(?:src|href)="(https?://[^"]+)"', html)
        assert not externals, f"{url} loads external resources: {externals}"


def test_contact_email_present() -> None:
    for url, path in PAGES.items():
        html = path.read_text(encoding="utf-8")
        assert CONTACT_EMAIL in html, f"{url} is missing the contact email"


def test_effective_date_placeholder_is_flagged() -> None:
    """Privacy and terms must keep the marked publication-date placeholder."""
    for url in ("/privacy", "/terms"):
        html = PAGES[url].read_text(encoding="utf-8")
        assert 'data-launch-blocker="effective-date"' in html, (
            f"{url} lost the launch-blocker marker"
        )


def test_privacy_clarifies_local_voice_transcription() -> None:
    html = PAGES["/privacy"].read_text(encoding="utf-8")
    assert "executed within our own application infrastructure" in html
    assert "external transcription API" in html


def test_no_secrets_in_website() -> None:
    """Guard against accidentally shipping credentials in the public site."""
    pattern = re.compile(
        r"(EAAG[a-zA-Z0-9]{20,}|whatsapp_token\s*=|access_token=|Bearer\s+[A-Za-z0-9])",
        re.IGNORECASE,
    )
    for path in WEBSITE.rglob("*"):
        if path.is_file():
            assert not pattern.search(path.read_text(encoding="utf-8", errors="ignore")), (
                f"possible secret in {path}"
            )
