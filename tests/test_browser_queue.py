"""Tests for the local ingest queue the browser extension posts into. No network."""
from __future__ import annotations

from dataclasses import replace

from fastapi.testclient import TestClient

from src.api import create_app
from src.config import BrowserModeCfg, load_config
from src.storage.db import session_scope
from src.storage.models import BROWSER_SOURCE, Video, VideoStatus

TOKEN = "test-token"
REEL = "https://www.instagram.com/reel/AbCdE12/"


def _enabled(cfg, **overrides):
    browser = BrowserModeCfg(enabled=True, ingest_token=TOKEN, **overrides)
    return replace(cfg, browser_mode=browser)


def _client(cfg) -> TestClient:
    return TestClient(create_app(cfg))


def _post(client: TestClient, urls: list[str], token: str = TOKEN):
    return client.post("/queue/urls", json={"urls": urls}, headers={"X-Ingest-Token": token})


def test_queue_routes_absent_when_browser_mode_is_off(tmp_cfg, db):
    client = _client(tmp_cfg)

    assert _post(client, [REEL]).status_code == 404
    assert client.get("/queue").status_code == 404


def test_queue_rejects_a_wrong_token(tmp_cfg, db):
    response = _post(_client(_enabled(tmp_cfg)), [REEL], token="nope")

    assert response.status_code == 401
    with session_scope() as session:
        assert session.query(Video).count() == 0


def test_queue_is_unavailable_without_a_configured_token(tmp_cfg, db):
    cfg = replace(tmp_cfg, browser_mode=BrowserModeCfg(enabled=True, ingest_token=""))

    response = _post(_client(cfg), [REEL], token="")

    assert response.status_code == 503


def test_queue_accepts_reels_and_normalizes_urls(tmp_cfg, db):
    client = _client(_enabled(tmp_cfg))

    response = _post(client, ["https://instagram.com/reel/AbCdE12?igsh=xyz"])

    assert response.status_code == 200
    assert response.json() == {"accepted": 1, "duplicates": 0, "queue_size": 1}
    with session_scope() as session:
        video = session.query(Video).one()
    assert (video.source, video.source_id, video.page_url) == (BROWSER_SOURCE, "AbCdE12", REEL)
    assert video.status == VideoStatus.DISCOVERED


def test_queue_accepts_a_plain_post_url_and_keeps_its_form(tmp_cfg, db):
    client = _client(_enabled(tmp_cfg))

    response = _post(client, ["https://www.instagram.com/p/DKKowgLsyJr/"])

    assert response.status_code == 200
    with session_scope() as session:
        video = session.query(Video).one()
    assert (video.source_id, video.page_url) == (
        "DKKowgLsyJr",
        "https://www.instagram.com/p/DKKowgLsyJr/",
    )


def test_queue_treats_the_same_shortcode_under_p_and_reel_as_one_post(tmp_cfg, db):
    client = _client(_enabled(tmp_cfg))
    _post(client, [REEL])

    response = _post(client, ["https://www.instagram.com/p/AbCdE12/"])

    assert response.json() == {"accepted": 0, "duplicates": 1, "queue_size": 1}


def test_queue_counts_a_repeated_url_as_a_duplicate(tmp_cfg, db):
    client = _client(_enabled(tmp_cfg))
    _post(client, [REEL])

    response = _post(client, [REEL])

    assert response.json() == {"accepted": 0, "duplicates": 1, "queue_size": 1}


def test_queue_refuses_a_batch_containing_a_non_reel_url(tmp_cfg, db):
    client = _client(_enabled(tmp_cfg))

    response = _post(client, [REEL, "https://example.com/watch?v=1"])

    assert response.status_code == 422
    with session_scope() as session:
        assert session.query(Video).count() == 0  # the whole batch is refused


def test_queue_stops_accepting_when_full(tmp_cfg, db):
    client = _client(_enabled(tmp_cfg, max_queue_size=1))
    _post(client, [REEL])

    response = _post(client, ["https://www.instagram.com/reel/Zzzzz99/"])

    assert response.json() == {"accepted": 0, "duplicates": 0, "queue_size": 1}


def test_queue_listing_shows_only_pending_browser_rows(tmp_cfg, db):
    client = _client(_enabled(tmp_cfg))
    _post(client, [REEL, "https://www.instagram.com/reel/Zzzzz99/"])
    with session_scope() as session:
        session.query(Video).filter(Video.source_id == "Zzzzz99").one().status = (
            VideoStatus.DOWNLOADED
        )

    body = client.get("/queue").json()

    assert body["total"] == 1
    assert body["items"][0]["source_id"] == "AbCdE12"


BROWSER_YAML = """
app: {name: test-app}
storage: {database: "sqlite:///data/test.db"}
collectors:
  ytdlp: {enabled: true}
browser_mode:
  enabled: true
  ingest_token: "${BROWSER_INGEST_TOKEN}"
  max_queue_size: 7
processing:
  video: {min_duration: 5, max_duration: 60}
export: {}
"""


def test_browser_mode_reads_the_token_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("BROWSER_INGEST_TOKEN", "from-env")
    path = tmp_path / "config.yaml"
    path.write_text(BROWSER_YAML)

    cfg = load_config(path)

    assert cfg.browser_mode.enabled is True
    assert cfg.browser_mode.ingest_token == "from-env"
    assert cfg.browser_mode.ingest_token_env == "BROWSER_INGEST_TOKEN"
    assert cfg.browser_mode.max_queue_size == 7
