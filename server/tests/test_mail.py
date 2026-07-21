"""Transactional email (server.md#email).

Covers what the endpoint tests can't see: the built message, the link, the
no-key dev path, and the guarantee that a provider failure never propagates.
"""

import httpx
import pytest

from app import config, mail


@pytest.fixture
def configured(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("IMAGEGENIE_RESEND_API_KEY", "test-key")
    monkeypatch.setenv("IMAGEGENIE_APP_BASE_URL", "https://app.test/")  # trailing slash
    monkeypatch.setenv("IMAGEGENIE_MAIL_FROM", "genie@imagegenie.dev")
    config.get_settings.cache_clear()
    yield
    mail.reset_mail_sender()
    config.get_settings.cache_clear()


@pytest.fixture
def outbox(configured) -> list:
    captured: list[tuple] = []
    mail.set_mail_sender(
        lambda email, from_address, key: captured.append((email, from_address, key))
    )
    return captured


def test_verification_email_carries_a_usable_link(outbox: list) -> None:
    mail.send_verification_email("labeler@imagegenie.dev", "tok123")
    email, from_address, api_key = outbox[0]

    assert email.to == "labeler@imagegenie.dev"
    assert "Confirm" in email.subject
    # Single slash — the base URL's trailing slash must not double up.
    assert "https://app.test/verify-email?token=tok123" in email.text
    assert "https://app.test/verify-email?token=tok123" in email.html_body
    assert from_address == "genie@imagegenie.dev"
    assert api_key == "test-key"


def test_invite_email_links_to_signup_with_the_address(outbox: list) -> None:
    mail.send_invite_email("newbie@imagegenie.dev", "admin@imagegenie.dev")
    email = outbox[0][0]
    assert "https://app.test/signup?email=newbie@imagegenie.dev" in email.text
    assert "admin@imagegenie.dev" in email.text  # who invited them


def test_html_bodies_escape_interpolated_values(outbox: list) -> None:
    """Addresses are validated upstream, but escaping at the boundary is what
    stays correct if that validation ever loosens."""
    mail.send_invite_email("victim@x.dev", '<script>alert(1)</script>')
    email = outbox[0][0]
    assert "<script>" not in email.html_body
    assert "&lt;script&gt;" in email.html_body


def test_both_plain_text_and_html_are_sent(outbox: list) -> None:
    mail.send_verification_email("someone@x.dev", "tok")
    email = outbox[0][0]
    assert email.text and email.html_body
    assert "<" not in email.text  # the text part is genuinely plain


def test_without_an_api_key_nothing_is_sent_and_the_link_is_logged(
    monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    monkeypatch.delenv("IMAGEGENIE_RESEND_API_KEY", raising=False)
    config.get_settings.cache_clear()
    sent: list = []
    mail.set_mail_sender(lambda *args: sent.append(args))

    with caplog.at_level("WARNING", logger="app.mail"):
        mail.send_verification_email("dev@local.test", "tok999")

    assert sent == []  # never reaches the provider
    assert "tok999" in caplog.text  # but a developer can still complete the flow
    mail.reset_mail_sender()
    config.get_settings.cache_clear()


def test_a_provider_failure_never_propagates(outbox: list, caplog) -> None:
    """Delivery is best-effort: the account already exists by the time we send,
    so a mail outage must not turn a successful signup into an error."""
    def exploding_sender(*_args):
        raise httpx.ConnectError("resend unreachable")

    mail.set_mail_sender(exploding_sender)
    with caplog.at_level("ERROR", logger="app.mail"):
        mail.send_verification_email("someone@x.dev", "tok")  # must not raise
    assert "delivery failed" in caplog.text


def test_transport_posts_the_expected_payload(configured, monkeypatch) -> None:
    """The one thing the seam can't cover — the real Resend call shape."""
    captured: dict = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return httpx.Response(200, request=httpx.Request("POST", url))

    monkeypatch.setattr(mail.httpx, "post", fake_post)
    mail.resend_sender(
        mail.OutgoingEmail(to="a@b.dev", subject="Subj", text="plain", html_body="<p>rich</p>"),
        "from@imagegenie.dev",
        "secret-key",
    )

    assert captured["url"] == mail.RESEND_ENDPOINT
    assert captured["headers"]["Authorization"] == "Bearer secret-key"
    assert captured["json"] == {
        "from": "from@imagegenie.dev",
        "to": "a@b.dev",
        "subject": "Subj",
        "text": "plain",
        "html": "<p>rich</p>",
    }
    # server.md#request-resilience: every outbound request carries a timeout.
    assert captured["timeout"] == mail.SEND_TIMEOUT_SECONDS


def test_transport_raises_on_a_non_2xx(configured, monkeypatch) -> None:
    """`resend_sender` must raise so `_deliver` can log it — silence here would
    make a provider rejection invisible."""
    def fake_post(url, **_kwargs):
        return httpx.Response(422, request=httpx.Request("POST", url))

    monkeypatch.setattr(mail.httpx, "post", fake_post)
    with pytest.raises(httpx.HTTPStatusError):
        mail.resend_sender(
            mail.OutgoingEmail(to="a@b.dev", subject="s", text="t", html_body="h"),
            "from@imagegenie.dev",
            "key",
        )
