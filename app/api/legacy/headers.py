from collections.abc import Callable

from fastapi import Response


def deprecated(successor: str) -> Callable[[Response], None]:
    """Dependency that marks a legacy route as deprecated and points to its v1 replacement (RFC 9745 style)."""

    def mark(response: Response) -> None:
        response.headers["Deprecation"] = "true"
        response.headers["Link"] = f'<{successor}>; rel="successor-version"'

    return mark
