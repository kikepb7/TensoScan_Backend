## Spec y tareas
<!-- Spec (B-00x) y tareas (T-B00x-yy) que implementa este PR -->

## Cambios
<!-- Qué cambia y por qué -->

## Pruebas
<!-- Tests añadidos o actualizados -->

## Checklist (Definition of Done)
- [ ] `uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest` en verde
- [ ] Los routers no acceden a Mongo ni capturan `Exception`; los errores son `DomainError`
- [ ] No se registran contraseñas, tokens ni valores de medición
- [ ] Contratos actualizados (`specs/contracts`) y copiados a la app si cambian
- [ ] Fechas con zona horaria (`datetime.now(UTC)`)
