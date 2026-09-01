## What this changes

<!-- One paragraph: the user-visible behaviour before and after. Link the issue if there is one. -->

## Why

<!-- The defect or need, with evidence (a log line, a screenshot, a measurement). -->

## How it was verified

- [ ] `./backend/venv/Scripts/python.exe -m ruff check .` clean
- [ ] `./backend/venv/Scripts/python.exe -m pytest backend` green (counts: …)
- [ ] `npm --prefix frontend run test -- --watch=false` green (counts: …)
- [ ] UI changes checked in the browser at the breakpoints claimed (screenshot per claim)

## Public surface

<!-- Any new or changed route, id, slug, schema key, serialized order, env var or exit code. Released surfaces are frozen: deprecate or alias, never rename. "None" is a valid answer. -->

## Notes for the reviewer

<!-- Anything you would want to know before reading the diff: what was deliberately left out, what you are unsure about. -->
