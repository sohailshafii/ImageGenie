"""Transactional email via Resend (server.md#email).

One thin transport plus a per-message builder, mirroring the ChatApp reference.
Resend is called over plain HTTP rather than through its SDK — the API is a
single POST, and a dependency that wraps one request isn't worth carrying.

Three properties matter here:

- **Sending never breaks a flow.** Delivery failures are logged, never raised.
  The account already exists at that point and the user can ask for a resend;
  failing the request instead would strand a created account behind an error.
- **No API key means log the link, don't send.** That keeps local dev working
  with no credentials. It also writes a token-bearing link into the logs, so it
  is strictly a development affordance (see the warning in `_deliver`).
- **The sender is swappable** (`set_mail_sender`) so tests assert on the built
  message — subject, recipient, and the link — instead of skipping the builder
  entirely because no key is configured.
"""

from __future__ import annotations

import html
import logging
from collections.abc import Callable
from dataclasses import dataclass

import httpx

from .config import get_settings

logger = logging.getLogger(__name__)

RESEND_ENDPOINT = "https://api.resend.com/emails"
# server.md#request-resilience requires a timeout on every outbound request; a
# hung mail provider must not hold a request open indefinitely.
SEND_TIMEOUT_SECONDS = 10.0


@dataclass(frozen=True)
class OutgoingEmail:
    to: str
    subject: str
    text: str
    html_body: str


MailSender = Callable[[OutgoingEmail, str, str], None]


def resend_sender(email: OutgoingEmail, from_address: str, api_key: str) -> None:
    """POST one message to Resend. Raises on a non-2xx response."""
    response = httpx.post(
        RESEND_ENDPOINT,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "from": from_address,
            "to": email.to,
            "subject": email.subject,
            "text": email.text,
            "html": email.html_body,
        },
        timeout=SEND_TIMEOUT_SECONDS,
    )
    response.raise_for_status()


_sender: MailSender = resend_sender


def set_mail_sender(sender: MailSender) -> None:
    """Swap the transport — the test seam."""
    global _sender
    _sender = sender


def reset_mail_sender() -> None:
    global _sender
    _sender = resend_sender


def _deliver(email: OutgoingEmail, dev_log_line: str) -> None:
    """Send `email`, or log `dev_log_line` when no API key is configured.

    Best-effort by contract: a provider failure is logged and swallowed.
    """
    settings = get_settings()
    if not settings.resend_api_key:
        # TODO(email): dev-only path — this writes a token-bearing link into the
        # logs. Any deployed environment must set IMAGEGENIE_RESEND_API_KEY.
        logger.warning("RESEND_API_KEY unset — not sending. %s", dev_log_line)
        return
    try:
        _sender(email, settings.mail_from, settings.resend_api_key)
        logger.info("sent %r to %s", email.subject, email.to)
    except Exception:
        # Deliberately broad: no delivery failure may propagate into the request.
        logger.exception("email delivery failed for %s", email.to)


def _link(path: str, query: str) -> str:
    return f"{get_settings().app_base_url.rstrip('/')}{path}?{query}"


def send_verification_email(recipient: str, token: str) -> None:
    """Email the one-time confirmation link for a new account."""
    link = _link("/verify-email", f"token={token}")
    _deliver(
        OutgoingEmail(
            to=recipient,
            subject="Confirm your ImageGenie account",
            text=(
                "Welcome to ImageGenie.\n\n"
                f"Confirm your email address by opening this link:\n\n{link}\n\n"
                "The link expires in 24 hours. If you didn't sign up, ignore this."
            ),
            # The recipient address is escaped before it reaches an HTML body —
            # it is validated upstream, but escaping at the boundary is the habit
            # that stays correct when validation changes.
            html_body=(
                "<p>Welcome to ImageGenie.</p>"
                f'<p>Confirm <strong>{html.escape(recipient)}</strong> by opening '
                f'<a href="{html.escape(link)}">this link</a>.</p>'
                "<p>The link expires in 24 hours. "
                "If you didn't sign up, ignore this email.</p>"
            ),
        ),
        dev_log_line=f"verification link for {recipient}: {link}",
    )


def send_invite_email(recipient: str, invited_by: str) -> None:
    """Email an admin-minted signup invitation."""
    link = _link("/signup", f"email={recipient}")
    _deliver(
        OutgoingEmail(
            to=recipient,
            subject="You've been invited to ImageGenie",
            text=(
                f"{invited_by} invited you to help label 3D models on ImageGenie.\n\n"
                f"Create your account here:\n\n{link}\n\n"
                "The invite expires in 14 days."
            ),
            html_body=(
                f"<p>{html.escape(invited_by)} invited you to help label 3D models "
                "on ImageGenie.</p>"
                f'<p><a href="{html.escape(link)}">Create your account</a>.</p>'
                "<p>The invite expires in 14 days.</p>"
            ),
        ),
        dev_log_line=f"invite for {recipient} by {invited_by}: {link}",
    )
