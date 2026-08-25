# 📡 API Conventions & Endpoint Standards

<system_directives>
This document defines the naming conventions, versioning strategy, and error/response schemas for the FastAPI backend. It describes what IS implemented — where a convention is aspirational, it is marked **not implemented**.
</system_directives>

<endpoint_naming>
1. **[URL STRUCTURE]** Endpoints follow RESTful conventions:
   - **Plural nouns** for collections: `/datasets`, `/jobs`, `/projects`
   - **kebab-case** for multi-word segments: `/caption-variant`, `/resume-from-checkpoint`
   - Nest sub-resources logically: `/datasets/{name}/caption-variants`, `/jobs/{job_id}/checkpoints`
2. **[HTTP METHODS]**
   - `GET` — read (list, detail). Idempotent.
   - `POST` — create, or an action that triggers processing (`/datasets/{name}/scan`, `/jobs/{id}/start`).
   - `PUT` — full replacement or a settings write (`/jobs/{id}/config`, `/jobs/settings/auto-resume`).
   - `PATCH` — partial update (`/datasets/{name}/video/trim`, control role patch).
   - `DELETE` — removal. Idempotent.
3. **[ACTION ENDPOINTS]** Non-CRUD operations use verb sub-paths: `/jobs/{id}/restart`, `/jobs/{id}/resume-from-checkpoint`, `/jobs/stats/recompute`, `/system/update/apply`.
</endpoint_naming>

<versioning_and_removal>
1. **[NO URL VERSIONING]** No `/v1/` prefix. The API evolves additively.
2. **[ROUTE REMOVAL — the real pattern]** A route is **removed outright once its last frontend caller is gone** — not soft-deprecated in place. The retirement of the `/api/settings` 410 migration guard (its stubs had zero callers left) is the reference: confirm no caller, delete the route (and its schema/test), done. There is no `Field(deprecated=True)` + 2-cycle-warning flow in this codebase; do not add one to satisfy an example.
</versioning_and_removal>

<request_response_schemas>
1. **[ALL SCHEMAS ARE PYDANTIC V2]** Every request body and response is a Pydantic `BaseModel` subclass; shared models live in `api/schemas/`. Raw `dict` returns are being eliminated.
2. **[response_model COVERAGE]** Routes declare `response_model=...` (the bulk already do; new routes MUST). This is what keeps the frontend's typed services honest and lets full-payload route tests catch field regressions.
3. **[OPEN MODEL for dynamic payloads]** When a response legitimately carries a variable/pass-through shape (rerun-config, template body, project extras), use an **open model** — `model_config = ConfigDict(extra="allow")` on the response model (see `project_routes.py`, `history_routes.py`, `template_routes.py`) — rather than falling back to a bare `dict`.
4. **[NAMING]** Request bodies: `<Resource>Create` / `<Resource>Update`. Responses: `<Resource>Response` / `<Resource>DetailResponse`.
5. **[PAYLOAD CARRIES ONLY WHAT CALLERS READ]** A list endpoint returns what a list view renders. Per-item detail (per-file metadata maps, nested blobs, anything the grid never displays) belongs on the detail route. Before adding a field to a LIST response, confirm a caller reads it; before keeping one, confirm the same. The largest responses in this API got that way one convenience field at a time.
6. **[RESPONSE FILTERING IS VERIFIED AGAINST THE BODY]** `response_model_exclude` addresses the *response model's* shape: on `response_model=list[X]` the key must be `{"__all__": {"field"}}`. The bare `{"field"}` form — correct for a single-object route — is a SILENT NO-OP on a list route and ships the field anyway. Never assert this in a test by inspecting the route's kwargs; assert the field is absent from an actual response body.
7. **[PAGINATION — current convention]** List endpoints take **bare `limit` / `offset` query params** (or none) and return the **full array** as `response_model=list[<Resource>Response]`. There is **no `PaginatedResponse` envelope** (`items` + `total` + `page`) implemented; if pagination metadata is ever needed, introducing such an envelope is a **future, not-yet-adopted** convention — do not assume callers expect it.
</request_response_schemas>

<error_responses>
1. **[ERROR ENVELOPE — implemented]** Errors are serialized by exception handlers in `main.py` into the `ErrorResponse` model (`api/schemas/common_schemas.py`):
   ```json
   {
     "detail": "Human-readable error message",
     "error_code": "NOT_FOUND",
     "context": {}
   }
   ```
   `detail` stays backward-compatible (existing clients that read only `detail` keep working); `error_code` + `context` are additive. `_error_code_for(status_code)` maps the status to the token. Handlers cover `StarletteHTTPException`, `RequestValidationError`, and the domain `DrainActive` error.
2. **[404 DEPENDENCY]** Dataset-scoped routes get their 404 from the shared raise-404 helper `dataset_or_404` in `api/_deps.py`. Each route module declares its own one-line `get_dataset_or_404(name)` dependency on top of it (`return dataset_or_404(dataset_manager.get_dataset(name))`) — the per-module copies are **deliberate, not duplication to clean up**: the wrapper reads its own module-level `dataset_manager`, which is what lets the test suite substitute a fake manager per route module (`@patch("app.api.<module>.dataset_manager")`). A single shared lookup dependency would freeze its own import and never observe those patches. Do not open-code `if not dataset: raise HTTPException(404, ...)`, and do not centralize the lookup.
3. **[HTTP STATUS CODES]** `400`/`422` validation · `404` not found · `409` conflict (duplicate/locked) · `403` feature unavailable (e.g. self-update off) · `500` unexpected (full traceback logged via structlog, never returned to the client).
</error_responses>

<async_patterns>
1. **[ASYNC-FIRST]** Route handlers are `async def`.
2. **[to_thread DISCIPLINE]** Blocking work (DB repo calls, file I/O, CPU-bound helpers) is wrapped in `asyncio.to_thread` so nothing blocks the event loop — including project CRUD, which routes its repo calls through `to_thread`.
3. **[LONG-RUNNING OPS → TASKS]** GPU-bound / long operations (ML inference, scanning, upscaling, video split/scene-detect) are offloaded to the **background task framework** and return a task id; progress streams over WebSocket.
4. **[WEBSOCKET UPDATES]** Real-time progress is pushed over `/ws`, not polled. Routes that trigger background work emit progress events.
</async_patterns>

<entity_events>
1. **[EMIT ON MUTATION]** State-changing routes emit a typed entity-change event via `entity_events.emit_entity_change` so the frontend stores stay in sync. `EntityName` currently covers `job`, `dataset`, `media_item`, `settings`, `registry_model`, `overlay`, `project`, `template`; `EntityOp` is `created` / `updated` / `deleted` / `bulk_deleted`.
2. **[DATASET FILE CHANGES]** Operations that change files on disk additionally broadcast `dataset.invalidated` so `DatasetSyncService` can replace-not-merge the affected stores.
</entity_events>
