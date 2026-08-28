# Security

MRLN Arcane Tuner is a **local-first** application: you run it on your own machine, it
trains on your own files, and by default nothing it does is reachable from anywhere else.
Most of this document is therefore not about attackers — it is about **what the program
does with your files, your keys and your network**, so you can decide whether that matches
what you expected.

Every claim below names the file that implements it. If a claim and the code disagree, the
code is right and this document is a bug — please report it as one.

## Reporting a vulnerability

Use **GitHub's private vulnerability reporting** on this repository (Security → Report a
vulnerability). That keeps the report private until a fix exists.

Please do not open a public issue for a security problem.

Include what you did, what happened, and what you expected. A proof of concept helps. This
is a solo-maintained project, so a fix is best-effort rather than a service-level promise:
you will get an acknowledgement, and you will be told honestly if something will not be
fixed.

## What runs where

The backend binds `127.0.0.1` by default and the frontend is served from the same origin.
In that configuration the application is reachable only from your own machine.

**If you bind it to an address other machines can reach, it refuses to start without a
shared access token.** Set `MRLN_AUTH_TOKEN` to a long random string and sign in once
(`backend/app/core/container_config.py`). This is a hard refusal rather than a warning,
because a training server exposed on a LAN with no token gives anyone on that network the
ability to read your datasets and run code on your GPU.

The container image is built from a pinned commit, and **nothing the application runs holds
root**: the entrypoint starts as root only long enough to `chown` a freshly-mounted data volume
(which arrives root-owned and would otherwise be unwritable), then drops to the app UID via
`setpriv` before exec'ing uvicorn (`Dockerfile`, `entrypoint.sh`). Stated precisely rather than as
"runs as non-root", because there is a moment when it is root and you should know what it is for.
If `setpriv` is missing the entrypoint warns loudly and continues as root rather than failing
silently — pinned by `backend/tests/test_container_hardening.py`.

## Your files

| What | Where it goes |
| --- | --- |
| Datasets, captions, masks | The dataset roots you choose. Captions are written beside the image as `<stem>.txt`, or under `captions/<definition>/` when a definition-scoped variant is active. |
| Trained LoRAs, checkpoints, samples | The output directory you choose (`./outputs` by default). |
| Model weights | The Hugging Face cache on your machine. |
| Application state | `settings.json` and a SQLite database (WAL mode) next to the backend. |

**Uploads are streamed to a temporary file with a 64 GiB ceiling**; anything larger is
refused with HTTP 413, and the temporary file is removed on every exit path including that
one (`backend/app/api/_upload_guard.py`).

**Routes that take a path from the URL are contained** so a request cannot walk out of the
directory it is scoped to (`backend/app/api/_path_guard.py`, pinned by
`backend/tests/api/test_path_traversal_containment.py` and six route-specific tests beside it).

Nothing is uploaded anywhere. There is no telemetry, no analytics and no crash reporting.

## Your keys

`settings.json` holds your Hugging Face token and any captioning-provider API keys. It is
written atomically, so an interrupted write cannot leave it truncated
(`backend/app/core/settings_manager.py`).

**Keys are masked in API responses** — the settings API returns `key_masked` and a
`key_set` boolean rather than the value (`backend/app/api/api_provider_routes.py`). They
are not written to logs.

`settings.json` is not encrypted. It is protected by your operating system's file
permissions and nothing else. Treat it as you would an SSH private key: do not commit it,
do not put it in a shared folder, and do not include it when sending someone a bug report.

## The network

Outbound connections happen only when you ask for them:

- **Hugging Face** — downloading model weights and tokenizers.
- **A captioning endpoint you name** — any OpenAI-compatible server. Locally this is
  deliberately unrestricted, because pointing at Ollama, LM Studio or vLLM on
  `localhost` is the entire point of the feature.
- **Ollama** — caption refinement, if you enable it.
- **This repository** — only when you run the in-app self-update, which is a `git pull`.

**The captioning endpoint is contained when the application runs hosted**
(`backend/app/core/url_guard.py`). In a container on rented infrastructure the same field
becomes a way to make the server issue requests to addresses the caller cannot reach —
other tenants, internal services, and above all the cloud metadata endpoint at
`169.254.169.254`, which hands credentials to anything that asks from inside the instance.
The guard is mode-dependent on purpose: local behaviour is unchanged, hosted is contained,
and someone genuinely running a provider beside their container can opt back in.

The application serves a **Content-Security-Policy**
(`backend/app/api/_security_headers.py`) and the frontend is same-origin, with fonts
self-hosted rather than fetched from a third party.

## Supported versions

This project is pre-1.0 and only the latest release is supported. There are no backported
security fixes for older versions — upgrade instead.

## Model weights are not covered by this policy

The application downloads and executes model weights from third parties under their own
licences. It cannot vouch for them. A model file is code as far as your GPU is concerned;
only load weights from sources you trust. See the licence and model-weight terms in the
[README](README.md).
