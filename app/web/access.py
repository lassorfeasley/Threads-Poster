"""Who sees what: the company's admins see the whole app, on every account;
customers see the streamlined flow: today's posts to approve, the calendar of
what's scheduled (where they can drag a post to another time), analytics, and
the "Your account" settings.

"Admin" is the role on a person's login (Supabase ``app_metadata.role``), not
a property of an account or a machine, so no account is special: the
climate_archives login is a customer like any other. A dashboard without
sign-in (the operator's own Mac) is the admin's by definition.

An admin can look through a customer's eyes (``/view-as?mode=customer``): the
same rules then apply to them, menus and server alike, until they switch back.

The server enforces the customer's limits; hiding links alone would leave
every admin page one typed URL away.
"""
from __future__ import annotations

import re

ADMIN_ROLE = "admin"
VIEW_COOKIE = "view_as"
CUSTOMER_HOME = "/inbox"
ONBOARDING_HOME = "/welcome"

# Whole sections a customer may use (the path itself and everything under it).
_CUSTOMER_SECTIONS = (
    "/welcome",
    "/inbox",
    "/calendar",
    "/analytics",
    "/schedule",
    "/brand",
    "/style-guide",
    "/subtitle-style",
    "/outro-card",
    "/connections",
    "/appearance",
    "/account",
    "/engagement/first-reply",
    "/media",
    "/static",
    "/brand-static",
    # Brand & audience shows the house-rule suggestions learned from edits.
    "/clip-engine/rule-proposals",
)

# Single endpoints: settings' front door, signing out, switching views, and
# connecting the account's own social profiles.
_CUSTOMER_PATHS = frozenset({
    "/settings",
    "/logout",
    "/view-as",
    "/threads/connect",
    "/instagram/connect",
    "/linkedin/connect",
    "/linkedin/callback",
    "/facebook/connect",
    "/facebook/callback",
    "/facebook/resume",
    "/x/connect",
    "/x/callback",
})


# Customer actions that live under admin sections: moving a queued post to
# another time slot by dragging it on the calendar.
_CUSTOMER_ACTIONS = (
    ("POST", re.compile(r"/post/\d+/pin-window")),
)

# Admin machinery that happens to live under a customer section.
_ADMIN_ONLY = frozenset({
    ("POST", "/inbox/settings"),     # the inbox's own switch and limits
    ("POST", "/schedule"),           # hand-edited posting windows (custom schedules)
    ("POST", "/schedule/cancel"),    # ...and canceling one
    ("POST", "/analytics/digest"),
    ("POST", "/welcome/reset"),      # starting an account's onboarding over
    ("POST", "/welcome/skip"),       # ...or marking it done without it
})

# What a customer can still open while their account isn't onboarded: the
# onboarding itself, what its pages load, signing out and the account page, and
# the connect flows (a later step sends them out to Meta and back).
_DURING_ONBOARDING = (ONBOARDING_HOME, "/static", "/brand-static", "/media", "/account")


def customer_may(path: str, method: str = "GET") -> bool:
    """Whether a customer may make this request."""
    if (method.upper(), path) in _ADMIN_ONLY:
        return False
    if any(method.upper() == m and rx.fullmatch(path) for m, rx in _CUSTOMER_ACTIONS):
        return True
    if path in _CUSTOMER_PATHS:
        return True
    return any(path == s or path.startswith(s + "/") for s in _CUSTOMER_SECTIONS)


def onboarding_applies(path: str, method: str = "GET") -> bool:
    """Whether an account still being onboarded sends this request to
    onboarding instead (customers only; the gate checks the account)."""
    if path in _CUSTOMER_PATHS and path != "/settings":
        return False
    return not any(path == s or path.startswith(s + "/") for s in _DURING_ONBOARDING)


def real_admin(*, sign_in_on: bool, role: str | None) -> bool:
    """An admin by who they are: anyone on a dashboard without sign-in, else a
    login with the admin role."""
    return (not sign_in_on) or (role or "").strip().lower() == ADMIN_ROLE


def sees_admin(*, real: bool, view: str | None) -> bool:
    """Whether this request gets the admin's app: a real admin who isn't
    looking through a customer's eyes."""
    return real and (view or "") != "customer"


def is_admin(request) -> bool:
    """For templates and routes. Defaults to the customer's view when the
    request never passed the sign-in check (public pages such as /login)."""
    return bool(getattr(request.state, "is_admin", False))
