"""REQ-B002-14: every request to a legacy route is logged (route and user id) to measure its remaining use.

REQ-B002-13 retires the legacy routes when fewer than 5 % of the measurement requests still use them for 14 days; these
lines are the data for that decision. Only the route template and the user id are logged (C-SEC-03).
"""

import logging

from fastapi import Request

from app.api.deps import CurrentUser

logger = logging.getLogger(__name__)


def _route_of(request: Request) -> str:
    route = request.scope.get("route")
    return f"{request.method} {getattr(route, 'path', request.url.path)}"


async def record_legacy_usage(request: Request, user: CurrentUser) -> None:
    route = _route_of(request)
    logger.info("Legacy route used: %s by user %s", route, user.id, extra={"legacy_route": route, "user_id": user.id})


async def record_anonymous_legacy_usage(request: Request) -> None:
    """For the legacy routes that run before there is a user (register and login)."""
    route = _route_of(request)
    logger.info("Legacy route used: %s", route, extra={"legacy_route": route, "user_id": None})
