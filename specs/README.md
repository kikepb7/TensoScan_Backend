# TensoScan Backend — Specs (Spec-Driven Development)

Fuente de verdad del desarrollo en la rama `recognition-refactor-feature`. Todo cambio de código
parte de una spec aprobada y su commit referencia la tarea (`T-B00x-yy`) y los requisitos
(`REQ-B00x-yy`).

## 1. Objetivo de la iniciativa

1. **Sacar el reconocimiento del servidor.** La app pasa a reconocer los dígitos en el
   dispositivo y en tiempo real (spec App A-002). El backend deja de recibir imágenes y de cargar
   TensorFlow, y pasa a recibir **mediciones confirmadas** por el usuario mediante una API v1
   tipada, validada e idempotente (B-002).
2. **Entrenar y publicar el modelo de dígitos que usará la app** (B-003). Vive en este
   repositorio, bajo `ml/`, aislado del servicio web.
3. **Refactorizar el servicio** para cumplir buenas prácticas de arquitectura, seguridad, testing
   y operación (B-001).

> **Plan de implementación y decisiones vigentes**: ver `TensoScan_App/specs/implementation-plan.md`
> (plan común a los dos repositorios). Resumen: app comercial → solo licencias permisivas en `ml/`;
> zona horaria `Europe/Madrid`; primero todo el código y después entrenamiento con datos reales
> (canal `beta` con 1 dispositivo, `production` con ≥ 3).

## 2. Mapa de specs

| ID | Spec | Tipo | Depende de | Estado |
|----|------|------|-----------|--------|
| 000 | [Auditoría del estado actual](000-baseline-audit/audit.md) | Análisis | — | ✅ Redactada |
| 001 | [Fundación de plataforma](001-platform-foundation/requirements.md) | Refactor | 000 | 👀 En revisión |
| 002 | [API de mediciones v1](002-measurements-api-v1/requirements.md) | Feature + migración | 001 | 🚧 En implementación |
| 003 | [Modelo de detección de dígitos](003-digit-recognition-model/requirements.md) | ML | — (en paralelo) | 👀 En revisión |

Contratos compartidos con la app (**fuente canónica aquí**):

| Contrato | Productor | Consumidor |
|----------|-----------|------------|
| [`contracts/api-v1.md`](contracts/api-v1.md) | B-001, B-002 | App A-002, A-003 |
| [`contracts/reading-rules.md`](contracts/reading-rules.md) + [`reading-golden-cases.json`](contracts/reading-golden-cases.json) | — (reglas compartidas) | App (validador y parser), B-002 (validación), B-003 (evaluación) |
| [`contracts/digit-detector-model.md`](contracts/digit-detector-model.md) | B-003 | App A-002 (fase E) |

**Regla de sincronización**: un cambio de contrato se hace aquí, sube la versión del documento y
se copia a `TensoScan_App/specs/contracts/` en un PR enlazado. El CI de cada repositorio ejecuta
los casos dorados.

## 3. Hoja de ruta y dependencias con la app

```
B-001 Fundación ──► B-002 API v1 ──────────────► (App A-002 fase D: sincronización)
                         └─ convivencia legacy ──► retirada tras publicar A-002/A-003
B-003 Modelo (en paralelo desde el día 1) ───────► (App A-002 fase E: LiteRT)
```

- B-003 empieza **en paralelo**: el generador sintético y el harness de evaluación no dependen de
  nada, y el cuello de botella es la captura de datos reales.
- La app no queda bloqueada por B-003 gracias al detector ML Kit (App ADR-A002-03).

## 4. Flujo SDD con Claude Code

1. **Specify** (`requirements.md`) → revisión humana.
2. **Design** (`design.md` con ADR).
3. **Tasks** (`tasks.md`).
4. **Implement**: una tarea cada vez, con los tests primero. Antes de marcar `[x]` se ejecuta:
   `uv run ruff check . && uv run ruff format --check . && uv run mypy app && uv run pytest`.
5. Commit: `T-B00x-yy | <descripción en inglés>`.

Estados de una spec: `📝 Borrador` → `👀 En revisión` → `✅ Aprobada` → `🚧 En implementación` → `🏁 Completada`.
