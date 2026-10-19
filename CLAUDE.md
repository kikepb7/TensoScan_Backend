# TensoScan Backend — Guía para Claude Code

Este repositorio se desarrolla con **Spec-Driven Development**. La fuente de verdad es `specs/`.

## Antes de escribir código
1. Lee `specs/README.md` y `specs/constitution.md`.
2. Localiza la tarea en el `tasks.md` de la spec activa (B-001, B-002 o B-003) y lee los `REQ-*`
   y las secciones de `design.md` que referencia.
3. Si tocas la API, las reglas de lectura o el modelo, revisa `specs/contracts/`. **Esta es la
   fuente canónica**: cualquier cambio sube la versión del contrato y se copia a
   `TensoScan_App/specs/contracts/`.

## Al implementar
- Una tarea cada vez, en orden. Nada fuera de spec (si falta algo, propón primero el cambio de
  spec).
- Capas: `api → application → domain ← infrastructure` (C-ARCH-01). Los routers no acceden a Mongo.
- Errores: lanza `DomainError`. Solo `app/core/errors.py` traduce a HTTP. Nunca uses
  `except Exception` en routers ni devuelvas `str(e)`.
- Fechas siempre *aware* en UTC (`datetime.now(UTC)`).
- `ml/` es un paquete aparte: el servicio web no lo importa.

## Verificación (Definition of Done)
```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy app && uv run pytest
# ML:
cd ml && uv run ruff check . && uv run pytest
```
- Marca la tarea con `[x]` en `tasks.md`.
- Commit: `T-<spec>-<nn> | <imperative description in English>`.

## Recordatorios críticos
- Nunca se reciben ni se guardan imágenes de usuario en v1 (C-SEC-05).
- La validación de mediciones = `reading-rules.md`, probada con `reading-golden-cases.json`.
- Los datos reales de ML no se suben a git; solo los manifiestos.
