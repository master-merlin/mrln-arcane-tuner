# MRLN Arcane Tuner — frontend

The Angular single-page app for MRLN Arcane Tuner. It is served by the FastAPI
backend in production; the dev server below exists only for frontend work.

**You do not need this directory to run the app.** For installing and running
MRLN Arcane Tuner, see the [root README](../README.md). Read on only if you are
changing the UI.

## Requirements

- Node 24 LTS (the version the project is built and tested against)
- A running backend — the SPA is not useful on its own. See the root README.

## Setup

```bash
npm ci
```

Use `npm ci` rather than `npm install`: it installs exactly what
`package-lock.json` pins, which is what CI and every other developer resolve to.

## Development server

```bash
npm start
```

Serves on `http://localhost:4200` and proxies `/api` and `/media` to the backend
on port 8000 (see `proxy.conf.json`). The page reloads on source changes.

Note: `localhost` resolves to IPv6 first on Windows. If the dev server appears to
start but the app cannot reach the backend, try `http://127.0.0.1:4200`.

## Unit tests

```bash
npm test -- --watch=false
```

Vitest, in a jsdom environment. `--watch=false` is what CI and the project's
verification gate run; omit it for a watching run while developing.

To run a single spec file:

```bash
npm test -- --watch=false --include="**/websocket.service.spec.ts"
```

## End-to-end tests

```bash
npm run e2e          # headless
npm run e2e:ui       # Playwright's interactive UI mode
```

Playwright, configured in `e2e/playwright.config.ts`. These drive a real browser
and need a running backend.

## Build

```bash
npm run build
```

Output goes to `dist/`. Production builds are optimised; bundle-size budgets are
configured in `angular.json`.

## Conventions

- **Standalone components only** — no `NgModule`s.
- **Signals** over RxJS where a signal will do.
- **New control flow** — `@if` / `@for` / `@switch`, never `*ngIf` / `*ngFor`.
- **Zoneless change detection.** Tests use `TestBed.tick()` and fake timers
  rather than `fakeAsync`.
- **Every value interpolated into a URL path is `encodeURIComponent`-encoded**,
  including into backend routes that accept slashes. A raw `#` or `?` in a
  filename silently truncates the URL. This is enforced by a test.
- **Fonts are self-hosted** in `public/fonts/` (SIL OFL 1.1, see `OFL.txt`
  there). Do not add a third-party font or CDN link — the app is deliberately
  same-origin, and a test enforces that too.

## `runtime-config.json`

`public/runtime-config.json` is fetched at startup. Every URL the app builds is
origin-relative, so its `backendPort` and `frontendPort` keys are **deprecated
and unused** — they are still accepted so that an existing deployment's file
keeps working, and they may be removed in a later release. Unknown keys and
out-of-range values are ignored rather than adopted.
