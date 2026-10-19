"""Layering rules of the constitution (C-ARCH-01), enforced by reading the imports of every module."""

import ast
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[2] / "app"

FRAMEWORKS = {
    "fastapi",
    "starlette",
    "pymongo",
    "bson",
    "motor",
    "jwt",
    "httpx",
    "pwdlib",
    "pydantic_settings",
    "uvicorn",
}

RULES = {
    "domain": FRAMEWORKS | {"app.application", "app.infrastructure", "app.api", "app.core"},
    "application": FRAMEWORKS | {"app.infrastructure", "app.api", "app.core"},
    "infrastructure": {"app.api"},
    "core": {"app.api", "app.infrastructure", "app.application"},
}


def _imports(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
    return found


def _violations(layer: str) -> list[str]:
    forbidden = RULES[layer]
    problems = []
    for path in sorted((APP / layer).rglob("*.py")):
        if "legacy" in path.parts[-2]:  # legacy_ocr / legacy_reports are retired in B-002
            continue
        for module in _imports(path):
            if any(module == rule or module.startswith(rule + ".") for rule in forbidden):
                problems.append(f"{path.relative_to(APP.parent)} imports {module}")
    return problems


@pytest.mark.parametrize("layer", sorted(RULES))
def test_layer_does_not_import_what_it_must_not_know(layer: str) -> None:
    assert _violations(layer) == []


def test_routers_never_touch_the_database_driver_directly() -> None:
    offenders = [
        str(path.relative_to(APP.parent))
        for path in (APP / "api").rglob("*.py")
        if "legacy" not in path.name and {"pymongo", "bson"} & {m.split(".")[0] for m in _imports(path)}
    ]
    assert offenders == []
