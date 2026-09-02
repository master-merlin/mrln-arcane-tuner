# Training

The Training screen is where a curated dataset and a saved way of working
turn into a queued job. It assumes you already have a dataset you trust (see
[`docs/datasets-guide.md`](datasets-guide.md)) and, ideally, a template for
this client and model already set up in a project (see
[`docs/projects-guide.md`](projects-guide.md) and
[`docs/templates-guide.md`](templates-guide.md)) — the Training screen is also
where those training templates get created and edited in the first place.
If you just want to fire off a run without touring the full form, a project's
**Quick Train** tab does the same job in three clicks — see the Projects
guide's Quick Train section; this page covers the full form it can hand off
to.

## What you can actually do with it

- Pick a model **family and definition**, apply a saved **training
  template** for it, and see every downstream field (LoRA parameters,
  optimizer, VRAM knobs, sampling) update to match.
- Attach one or more project-scoped **datasets** as training concepts, each
  with its own caption prefix, caption-dropout rate, repeat count and
  masking behavior — a single run can train on several datasets with
  different weighting.
- Read a **live VRAM estimate**, calibrated from your own past runs once
  you have some, before you queue anything — including a peak/available bar,
  a per-component breakdown, and any warnings the engine has about your
  configuration.
- Turn on **Adaptive Layer Targeting** so the run itself narrows which LoRA
  modules keep learning as training progresses, without hand-picking layers.
- Fine-tune training dynamics, LoRA rank/alpha, the optimizer, quantization,
  block-swap offloading, and sampling previews — every field the model's own
  schema declares, grouped and labeled exactly as the form shows them.
- Queue the run with **Start Training Session**, then track it on the Jobs
  screen.

## The shape of the screen

Three columns: a left **table of contents** that jumps to each schema-driven
group and tracks which one you're scrolled to (Template Selection is a fixed
card above the groups, not a TOC entry itself; a group like Expert Features
or Video Settings appears in the TOC only when the current model definition
declares fields for it); a center **form** built
from the model definition's own schema — every model family declares the
same base schema plus family-specific fields, so the form is identical in
structure across families and only its options and defaults change; and a
right **Estimate rail** with the live VRAM report. The launch bar sits above
the form and stays pinned to the bottom edge of the viewport as you scroll,
carrying a compact echo of wall time and peak VRAM plus the **Start Training
Session** button — reachable without scrolling back up — and, if any section
has an invalid field, a count with a jump-to-first-invalid shortcut.

![Training screen — TOC, Model Selection, Template Selection and the Estimate rail](images/training-screen-overview.png)

## Set up the run

### Template Selection

The first card on the form. The **Settings template** dropdown applies any
training template for the currently selected model definition — factory
defaults marked `(Default)`, or one of your own project/global templates.
**Clone as New Template**, **Rename** and **Delete** work exactly as on the
`/templates` screen (rename/delete are disabled on a default template);
editing a non-default template's fields autosaves as you type, and editing a
default template instead creates (or reuses) a per-definition "Default by
User" copy scoped to the current project, so the factory template itself is
never overwritten. **Export** and **Import** move a single training template
as a `.template.zip` without leaving the form. See
[`docs/templates-guide.md`](templates-guide.md) for the full template system
— branching, the `/templates` library, and import plans.

### Model Selection

Choose the **family** and, once a family is picked, the **definition** — the
concrete checkpoint from that family's YAML. A family is an architecture with
its own loader, driver, trainer, sampler and saver; a definition is one
shipped checkpoint of it. The app ships **29 families, 54 definitions**
across image, video and audio, but the dropdown only enumerates the
definitions that have cleared their availability gate — today that's **28
families, 51 definitions**; the three gated ones (currently the MiniMax H3
definitions) stay hidden until they clear. Also on this card: the
**quantization backend** and **quantization** level for the base model, and
the same pair for the text encoder — quantizing either trades some quality
for VRAM, and the base-model quantization can also mean training
acceleration on newer GPUs depending on the level chosen. A badge next to the
model name flags a **Unified Transformer** architecture (different VRAM
characteristics than a diffusion model) and, when the model's weights come
from a local path instead of the Hub, a **LOCAL**/**SAFETENSORS**/**OFFLINE**
chip.

## Tune what this run needs

Below Model Selection and the VRAM Budget card, the rest of the form is
generated from the schema in the same grouping the form shows:

- **General Settings** — the dataset rows this run trains on (one per
  attached dataset, scoped to the current project when you're inside one):
  a **caption prefix**, **caption dropout rate** (the chance a caption is
  dropped entirely, which enables classifier-free guidance at inference), a
  **repeat count**, caption toggles (use captions at all, and whether to
  prefer the model-aware caption variant), and masking toggles (include
  masked variants, optionally regenerating them with a chosen opacity and a
  minimum probability of keeping the original) — multiple datasets can be
  attached to the same run, each weighted independently; this is the same
  picker documented from the dataset side in the Datasets guide's "Move on"
  section. Also on this card: LoRA filename (**LoRA Prefix** / **Suffix** /
  **Name**, the name field supporting `{placeholder}` substitution), a
  global trigger word, output directory, and data toggles (latent caching,
  horizontal/vertical flip augmentation).
- **Training Dynamics** — max steps, batch size, gradient accumulation,
  gradient checkpointing (VRAM for speed), checkpoint cadence and how many to
  keep, resume-from-checkpoint with cache re-use toggles, target
  **resolutions** for bucketing, and **bucketing mode** (`kohya`: one bucket
  per image; `multi`: an image appears in every qualifying bucket for more
  latent diversity), and **timestep sampling strategy** — including `radc`
  (see below). Adaptive Layer Targeting's own card (below) is anchored
  inside this group.
- **LoRA Parameters** — **network rank** (adapter capacity), **network
  alpha** (scaling factor), whether to train the text encoder alongside the
  main model, and a **target layers** picker for training only specific
  transformer blocks.
- **Optimizer Settings** — the optimizer algorithm (AdamW/AdamW8bit, Prodigy,
  ProdigyPlusSF, Lion, Adafactor, StableAdamW, SophiaH/SophiaG, Shampoo,
  RAdam, AdEMAMix), learning rate, weight decay, and the LR schedule.
- **Expert Features** — advanced knobs specific to whichever optimizer is
  selected (e.g. ProdigyPlusSF's cautious-update/OrthoGrad/FOCUS/SPEED
  toggles, Sophia's Hessian sampling settings, Adafactor's relative-step
  behavior) — only the fields relevant to the chosen optimizer are shown.
- **Advanced Engine** — EMA (with its decay rate), `min_snr_gamma`, noise
  offset, VAE/text-encoder offloading and unloading, and the minimum-free-VRAM
  fraction the sampler waits for before it will run.
- **Video Settings** — shown only for video-capable definitions: frame rate,
  frame count, frame stride, sliding-window/temporal-coverage controls for
  long clips, first-frame-conditioning probability, MoE expert-swap settings
  where the family has dual experts, and audio training toggles for
  families that support joint audio.
- **Sampling** — the sample prompts and their cadence (every N steps, skip
  the first N), inference steps, guidance scale, and seed used for the
  periodic preview images shown while a job runs.

Each section header shows a live one-line summary (e.g. LoRA rank/alpha, step
count and batch size, optimizer and LR) and an OK/attention chip, so you can
see a section's state without opening it.

### VRAM Budget

Sits right under Model Selection, before the schema-driven groups, because it
reads from the live estimate rather than from form fields directly. It
carries the same peak-VRAM hero tile as the right-hand rail plus an
**Advanced** panel for **block swapping** — moving a percentage of
transformer blocks to CPU between steps to trade speed for VRAM headroom, at
a granularity finer than the model-level quantization above.

### Adaptive Layer Targeting

A full-width card inside Training Dynamics, enabled by its own toggle.
Periodically measures which LoRA modules are still moving (an EMA-smoothed
norm of each module's effective weight change) and either **freezes** the
cold ones in place or **rebuilds** the adapter — checkpointing and
relaunching the same job with the optimizer rebuilt over only the parameters
still active, which reclaims optimizer-state VRAM you can spend on batch
size or resolution instead. This is a regularizer, not a speedup on its own:
the forward pass still runs every module (a frozen module's learned delta
stays part of the model), so freeze mode's savings are about training
capacity, not wall time — rebuild mode is the one that actually frees VRAM.

The ranking runs **per projection group**, not globally
(`backend/app/engine/core/optimization/adaptive_heat.py:117`, `select_active`):
every block's `to_v` is ranked only against other `to_v` projections, never
against `ff.gate` or another shape entirely. A raw weight-change norm is not
comparable across matrix shapes — under grouped-query attention a `to_v`
delta has an order of magnitude fewer elements than a feed-forward delta, so
one global ranking would retire an entire pathway (e.g. all attention
value/output projections) on shape alone rather than on whether it actually
stopped learning. Each group also keeps its own share of the floor below, so
no pathway can be frozen out entirely.

Pick a **preset** (the same Conservative / Balanced / Aggressive factory
presets, or any of your own) to seed **Warm-up** (share of training left
untouched before measurement starts), **Energy kept** (share of a group's
weight-change norm that must remain active), **Floor** (`min_active_pct`,
the minimum fraction of each projection group that can never be frozen),
**Heat smoothing** (the EMA factor applied to each measurement — higher
reacts slower), the **measurement interval** in steps, and the **action**
(Freeze vs Rebuild — Rebuild adds a **minimum shrink to rebuild** threshold so
a marginal narrowing doesn't trigger a relaunch by itself). Editing any knob
after selecting a read-only factory preset branches it into your own template
copy automatically. Presets only seed values at selection time — the job
stores the knobs themselves, so editing a preset afterward never changes a
run that already queued. See the Templates guide's "Adaptive" domain for how
these presets live in the template system.

### Timestep sampling — Resolution-Aware Dynamic Curriculum (RADC)

**What.** `timestep_sampling` (in Training Dynamics) picks which noise levels
the run trains on and how their probability is shaped: `logit_normal`
(default), `uniform`, `sigmoid`, `cosmap`, `mode`, `flux_shift`, `model_shift`,
or `radc`. Every mode but `radc` samples from a **fixed** distribution for the
whole run. `radc` — Resolution-Aware Dynamic Curriculum — is the one mode that
**moves**: it shifts a Gaussian sampling window across the noise range as the
run progresses (`backend/app/engine/strategies/timestep_sampling.py:93-129`).

**Why.** Early in training, the model has nothing yet — there is no detail to
refine, only structure and composition to learn, which lives in the
high-noise steps. Late in training structure is settled and what is left to
improve is texture and fine detail, which lives in the low-noise steps. A
fixed distribution spends the same attention on both phases throughout;
RADC spends it where the run actually needs it at that point in training.

**How.** Four knobs, all in Training Dynamics once `timestep_sampling` is set
to `radc` (`backend/app/engine/models/base.py:514-557`):
- **Noise focus at training start** (`radc_start`, default `0.8`) — where the
  sampling window centers at step 0 (`1.0` = pure high-noise/structure,
  `0.0` = pure low-noise/detail).
- **Noise focus at training end** (`radc_end`, default `0.2`) — where it
  centers by the final step; the run interpolates linearly between the two.
- **Curve width** (`radc_width`, default `0.5`) — how broad the sampling
  window is around that center (`0.1` = tightly focused, `1.0` = broad).
- **Resolution cross-influence** (`radc_res_influence`, default `0.15`) — for
  multi-resolution datasets, nudges the center further toward detail for
  high-resolution images late in the run and keeps low-resolution images on
  structure longer, since a small image has less detail to refine in the
  first place (`0` disables the effect).

**What to expect.** The sampling center moves smoothly from `radc_start`
toward `radc_end` over the whole run — there is no discrete phase switch to
watch for, no log line that says "curriculum stage 2 begins now". If a
multi-resolution dataset is attached, higher-resolution images will
increasingly draw their training timesteps from the low-noise end as the run
progresses; this is a training-time internal, not something the sample
previews will visibly show you step to step.

## Read the estimate before you start

The **Estimate** card at the top of the right rail (the same one Quick Train
uses) shows wall time, throughput, VRAM, output size and disk footprint, each
with a confidence sub-label — `calibrated · N runs` once the backend has
history for this model, `estimated · defaults` before that. If no local runs
exist yet, an **Update stats from history** button backfills calibration from
past jobs and re-estimates. Below it, the **VRAM report** card breaks the
estimate down by component (model weights, text encoder, activations,
optimizer state, headroom, …) as a proportional bar plus a legend, with any
backend warnings about the configuration listed underneath, and — once a
learning-rate schedule is known — a one-line LR schedule readout.

## Start the run

**Start Training Session**, in the sticky bar at the bottom of the form, is
disabled until every section is valid. Queuing a job hands the whole
configuration to the [Jobs screen](jobs-guide.md), where you watch it run,
pause, resume or soft-stop it, and download the finished LoRA once it
completes.

## Recipes

**First run for a new model on a project.** Model Selection → pick the
family and definition → General Settings → attach the project's
dataset → check the VRAM estimate → Start. Once it's tuned the way you like,
Template Selection → **Clone as New Template** so the next dataset for this
client and model starts from it instead of a blank form.

**Reuse a client's setup.** Template Selection → pick their template for this
model → General Settings → swap in the new dataset → Start. Nothing
else needs re-deciding — that's the same "a new dataset arrives" arc the
Projects guide ends on.

**Recover VRAM headroom on a long run.** Turn on Adaptive Layer Targeting,
set the action to **Rebuild**, and give it a **minimum shrink to rebuild**
that only triggers on a real narrowing — the reclaimed optimizer-state VRAM
can go toward a larger batch size or resolution on the next attempt.

## What survives an update

A queued or running job's configuration is a snapshot taken at Start —
editing the template afterward, or upgrading the app, never changes a job
already in the queue or in history. Training templates themselves live in
the same database as everything else and are untouched by an update.

## Where things live

The training form is driven by the model definition's own YAML plus the
shared base schema (`backend/app/engine/models/base.py`) — a new field on
either surfaces here automatically, with no frontend change needed. A queued
job's full configuration, step metrics, checkpoint locations and final LoRA
path are recorded in the app's database and surfaced on the Jobs screen.
