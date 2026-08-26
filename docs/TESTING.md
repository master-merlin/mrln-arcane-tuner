# 🧪 Testing Strategy

<system_directives>
This document describes how the Python/ML backend and the Angular frontend are actually tested in this repo.
You operate on a **Windows OS**. Follow the `<windows_workspace_hygiene>` text-redirection rules in `CLAUDE.md` for long/noisy test runs (redirect to `.agent\workdir\`, read, delete).
We do not guess; we prove.
</system_directives>

<philosophy>
1. **Deterministic Verification:** Every feature, ML transformation, and UI component is verified by a deterministic test.
2. **Coverage follows logic:** New logic implies new tests.
3. **Regression Guard:** The suite must pass before a task is "complete."
4. **Idempotency:** Tests clean up after themselves (DB teardown AND PyTorch VRAM clearing via `torch.cuda.empty_cache()` + `gc.collect()`).
</philosophy>

<backend_ml_testing>
**DOMAIN: Python 3.12 (`backend/` — FastAPI & PyTorch LoRA pipeline). ~2853 tests.**

### Running

Always invoke pytest through the venv, and run **both** test roots:

```
.\backend\venv\Scripts\python.exe -m pytest backend\tests backend\app\engine\tests
```

- `backend/pytest.ini` pins `addopts = --import-mode=importlib` — this tolerates same-named test modules in different packages (a missing `__init__.py` no longer aborts the whole collection).
- `backend/tests/conftest.py::pytest_collection_finish` **fails the run if 0 tests are collected**, so a silently-broken collection can never pass green again.
- For long runs, redirect: `... -m pytest backend\tests backend\app\engine\tests 2>&1 | Tee-Object -FilePath .agent\workdir\agent_pytest_out.txt`, then read + delete the file.

### Conventions in use

1. **Unit (PyTorch/ML — CRITICAL SAFETY):** Test tensor math, LoRA adapter injection, and data transforms with *small dummy tensors* (e.g. `torch.randn(2, 4, 64, 64)`). NEVER load full SD/Flux/WAN checkpoints in unit tests. Default to `device='cpu'`; when CUDA is used, free VRAM in teardown.
2. **Unit (FastAPI):** Endpoints are exercised with `TestClient` (`fastapi.testclient`). Routes are pinned with **full-payload response tests** so a schema/field regression fails loudly.
3. **Real-seam family contract tests:** The class of bug unit mocks historically hid (encode→forward + PEFT-sync seam) is caught by *real-seam* contract tests parametrized across the trainer/driver families — they run the actual seam, not a mock. When adding or changing a family, extend these rather than mocking the seam away.
4. **Sabotage-checked pins:** Precision/timestep-scale contracts (flow-match `[0,1000]`, no autocast collapse) are pinned by tests that would fail if the invariant were broken.
5. **DB / migrations:** `conftest.py` fixtures yield isolated SQLite state and roll back after each test; the integer-versioned migrations (`core/db/migrations.py`, through v17) have their own tests.
6. **Registry mutation:** Family-registry mutations use an autouse reset fixture (ltx2-style) so a test that registers a family cannot leak into the next.

> The repo does **not** use `respx` or `polyfactory`. Do not introduce them to satisfy an example — mock HTTP with the existing patterns and build payloads explicitly.
</backend_ml_testing>

<frontend_testing>
**DOMAIN: Angular 22 (`frontend/`, zoneless). ~968 unit tests + 19 e2e across 6 files.**

### Unit (Vitest, jsdom)

```
cd frontend
npx ng test --watch=false
```

- Runner is **Vitest** (jsdom), configured via `src/test-setup.ts` — there is no Karma / ChromeHeadless.
- Isolate one spec: `npx ng test --include "**/path-to-spec.spec.ts" --watch=false`.
- **Zoneless conventions** (app removed zone.js):
  - No `fakeAsync`. Flush change detection with `settle()` / `TestBed.tick()`.
  - RxJS `debounceTime` + fake timers: `vi.useFakeTimers({ toFake: [...,'Date'] })` — you MUST include `'Date'` or debounce reschedules forever.
  - Always restore in `afterEach` (`vi.useRealTimers()` / restore mocks).
- **Components:** test Signal state transforms (`signal()`, `computed()`) and interaction, not Angular's render engine.
- **Services:** mock HTTP with `HttpTestingController` via `provideHttpClientTesting()`.

### E2E (Playwright)

```
cd frontend
npm run e2e          # playwright test -c e2e/playwright.config.ts
npm run e2e:ui       # interactive
```

- Playwright config self-starts the app on port **:4300**; the backend is a fully **in-browser mock** (`e2e/fixtures/mock-backend.ts`) — no live server.
- **Invariant:** the mock's `unhandledApi` list must stay empty — an unmocked endpoint fails the boot spec, which is how stale mocks get caught.
- Specs live in `e2e/specs/`: `boot-datasets`, `dataset-editors`, `dataset-settings`, `jobs`, `mass-edit-overlays`, `training-config`.

### Selector policy (non-negotiable)

- NEVER select by CSS class or HTML tag.
- Every test target carries `data-testid="feature-name"`; e2e locators use `getByTestId('feature-name')` only. Cypress is not used.
</frontend_testing>

<self_correction_protocol>
**Execute strictly if a test fails.**

1. **STOP:** don't immediately rewrite code or guess.
2. **ISOLATE:** backend `pytest -k "<name>"`; frontend `ng test --include "**/spec.spec.ts" --watch=false`.
3. **ANALYZE:** read the redirected output from `.agent\workdir`, then delete it. Logic error (new code wrong) or expectation error (assertion stale)? Heuristic: if it passed before your change, assume your change broke it — don't force-pass the test. For a tensor shape mismatch, print exact shapes before fixing.
4. **FIX & VERIFY:** apply the surgical fix, re-run the isolated test, then the full suite to confirm no regression.
5. **CIRCUIT BREAKER:** if the same isolated test fails 3× consecutively, STOP, write a one-line summary to `.agent\workdir\stuck.md`, and ask the user (per `CLAUDE.md` `<agent_sop>`).
</self_correction_protocol>
