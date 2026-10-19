from collections.abc import Callable

from fastapi import Response


def deprecation_headers(successor: str) -> dict[str, str]:
    """Headers that mark a legacy route as deprecated and point to its v1 replacement (RFC 9745 style)."""
    return {"Deprecation": "true", "Link": f'<{successor}>; rel="successor-version"'}


def deprecated(successor: str) -> Callable[[Response], None]:
    """Dependency that adds ``deprecation_headers``. Routes that build their own ``Response`` add them themselves."""

    def mark(response: Response) -> None:
        response.headers.update(deprecation_headers(successor))

    return mark
