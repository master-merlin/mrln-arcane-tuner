/**
 * Boot-time API fixtures for the e2e harness.
 *
 * Shapes mirror the REAL interfaces the app consumes so the datasets screen
 * renders without runtime errors:
 *   - `Dataset`           → src/app/services/dataset.ts
 *   - `MpxDistribution`   → src/app/services/dataset.ts
 *   - cache-stats         → DatasetService.getCacheStats() return shape
 *   - `Project`           → src/app/services/project.service.ts
 *   - version             → sidebar GET `/` → { version }
 *
 * The two datasets are crafted so the H/C/M state pills render at different
 * coverage tiers (full / mid / low / none) and the six KPI tiles aggregate to
 * non-trivial values.
 */

/** GET /runtime-config.json */
export const runtimeConfig = {
    backendPort: 8000,
    frontendPort: 4200,
};

/** GET /api/system/version — sidebar's ngOnInit app-version probe (every page). */
export const version = { version: 'e2e' };

/** GET /api/projects — shell loads this on boot. Empty list keeps the scope
 *  switcher in Global, which is what the boot smoke exercises. */
export const projects: unknown[] = [];

/**
 * 12 JSON step lines for the running job so the Jobs KPI row renders (LANE-143
 * UAT shots). `progress` is in PERCENT: `latestMetrics` derives
 * total_steps = step / (progress/100) = 1200 / 0.24 = 5000, and requires
 * `status === 'training'` (shared/job-metrics.ts).
 */
export const runningJobStepLogs: string[] = Array.from({ length: 12 }, (_, i) => {
    const k = i + 1;
    return JSON.stringify({
        status: 'training',
        step: 100 * k,
        loss: 0.42 - 0.01 * k,
        step_time: 1.8,
        samples_per_sec: 2.2,
        eta: 6840,
        progress: Math.round((100 * k) / 50),
        resolution: '1024x1024',
    });
});

/**
 * GET /api/jobs — not fetched on the datasets boot path, but the Jobs screen
 * (Flow F) renders these directly into the queue's RUNNING/PENDING groups.
 * Two rows so `training-job-queue` renders both groups; see the Flow F
 * fixtures block near the bottom of this file for the rest of the jobs-screen
 * mocks (history / auto-queue / auto-resume / samples / checkpoints).
 */
export const jobs = [
    {
        id: 'job-e2e-running',
        plugin_id: 'standard',
        config: { definition_id: 'flux-dev', lora_name: 'nightfall-run' },
        status: 'running',
        status_label: '',
        created_at: Math.floor(Date.now() / 1000) - 600,
        started_at: Math.floor(Date.now() / 1000) - 600,
        pid: 4242,
        logs: runningJobStepLogs,
        warnings: [] as string[],
    },
    {
        id: 'job-e2e-pending',
        plugin_id: 'standard',
        config: { definition_id: 'flux-dev', lora_name: 'queued-lora' },
        status: 'pending',
        created_at: Math.floor(Date.now() / 1000) - 60,
        priority: 0,
        logs: [] as string[],
        warnings: [] as string[],
    },
];

/** GET /api/tasks — the Task Center / TaskStore hydrates the queued-task list
 *  on boot. Empty = no active background tasks. */
export const tasks: unknown[] = [];

/**
 * GET /api/datasets — the library list. Two rows with distinct readiness so
 * the H/C/M pills show different tiers and the KPI rail aggregates.
 *
 * Row "alpha": fully captioned + fully masked + harmonized + cached → all
 *   pills green (full).
 * Row "bravo": partial captions (mid), no masks (none), no harmonization,
 *   uncached → mixed pills.
 */
const NOW = Math.floor(Date.now() / 1000);

export const datasets = [
    {
        id: 'd-alpha',
        name: 'alpha',
        path: '/data/alpha',
        description: 'Fully prepped dataset',
        created_at: NOW - 3 * 86400,
        last_scanned_at: NOW - 3600,
        file_count: 100,
        total_size_bytes: 1_500_000_000,
        multimedia_count: 100,
        caption_count: 100,
        mask_count: 100,
        caption_coverage: true,
        harmonization_score: 1.0,
        classifier: 'portrait',
        version: '1.2.0',
        has_cache: true,
        trigger_word: 'alph',
        tags: ['ready', 'portrait'],
        notes: '',
        median_quality_score: 0.31,
        excluded_count: 0,
    },
    {
        id: 'd-bravo',
        name: 'bravo',
        path: '/data/bravo',
        description: 'Work in progress',
        created_at: NOW - 1 * 86400,
        last_scanned_at: NOW - 7200,
        file_count: 40,
        total_size_bytes: 600_000_000,
        multimedia_count: 40,
        caption_count: 25,
        mask_count: 0,
        caption_coverage: false,
        harmonization_score: 0,
        classifier: 'landscape',
        version: '0.1.0',
        has_cache: false,
        trigger_word: 'brav',
        tags: ['landscape'],
        notes: '',
        median_quality_score: 0.22,
        excluded_count: 2,
    },
];

/** GET /api/datasets/cache/stats */
export const cacheStats = {
    total_bytes: 800_000_000,
    latent_bytes: 500_000_000,
    embedding_bytes: 300_000_000,
    cached_datasets: 1,
    dataset_root_bytes: 2_100_000_000,
};

/** GET /api/datasets/stats/mpx-distribution */
export const mpxDistribution = {
    total_images: 140,
    avg_size_bytes: 15_000_000,
    avg_megapixels: 4.2,
    median_megapixels: 4.0,
    buckets: [
        { range_mp_min: 0, range_mp_max: 1, count: 5 },
        { range_mp_min: 1, range_mp_max: 2, count: 10 },
        { range_mp_min: 2, range_mp_max: 3, count: 20 },
        { range_mp_min: 3, range_mp_max: 4, count: 35 },
        { range_mp_min: 4, range_mp_max: 5, count: 40 },
        { range_mp_min: 5, range_mp_max: 6, count: 18 },
        { range_mp_min: 6, range_mp_max: 7, count: 8 },
        { range_mp_min: 7, range_mp_max: 8, count: 3 },
        { range_mp_min: 8, range_mp_max: 9, count: 1 },
        { range_mp_min: 9, range_mp_max: 10, count: 0 },
    ],
};

/* ════════════════════════════════════════════════════════════════════════
 * Training screen (Flow B) fixtures.
 *
 * Shapes mirror the REAL interfaces the training form consumes so the dynamic
 * config renderer actually builds the asserted `config-*` controls:
 *   - ModelDefinition  → src/app/screens/training-screen/training-screen.ts
 *   - SchemaNode       → src/app/components/training/schema-node.ts (custom
 *                        JSON-Schema dialect read by training-dynamic-config)
 *   - TrainingEstimate → src/app/services/job.ts (POST /jobs/estimate)
 *   - VRAMReport       → src/app/services/system.service.ts (estimate.vram)
 *   - ModelCapabilities→ src/app/services/model-capabilities.service.ts
 *   - Template         → src/app/services/template.service.ts
 *
 * The training screen, on mount:
 *   GET /api/models/definitions            → trainingModels
 *   GET /api/plugins/standard/schema?t=…   → trainingSchema (defs.length > 0)
 *   GET /api/models/settings               → modelSettings (browse default path)
 *   GET /api/templates/training            → trainingTemplates (empty → Default)
 *   GET /api/models/capabilities/{id}      → modelCapabilities (×2: form + card)
 *   GET /api/models/definitions/{id}/source→ modelSource (badge)
 *   POST /api/jobs/estimate                → trainingEstimate (debounced VRAM)
 *   GET  /config_help.json                 → served by ng (not /api), 200 [] ok
 * ════════════════════════════════════════════════════════════════════════ */

/**
 * GET /api/models/definitions — the training model picker list. Two defs in one
 * family; the form auto-selects the first (`flux-dev`) on load.
 */
export const trainingModels = [
    { id: 'flux-dev', name: 'FLUX.1 Dev', family: 'flux', architecture_params: {} },
    { id: 'flux-schnell', name: 'FLUX.1 Schnell', family: 'flux', architecture_params: {} },
];

/**
 * GET /api/plugins/standard/schema — a small but representative training schema.
 *
 * The dynamic form (training-dynamic-config) groups properties by their `group`
 * key (MODEL_SELECTION → hardcoded section; BASE/STRATEGY/NETWORK/OPTIMIZER →
 * collapsible cards) and renders one control per field keyed by `type`:
 *   - string + enum → <select data-testid="config-select-{key}">
 *   - string        → <input  data-testid="config-input-{key}">
 *   - integer/number→ <input  data-testid="config-input-{key}"> (number)
 *   - boolean       → <input  data-testid="config-checkbox-{key}">
 *   - array (prim.) → app-dynamic-form-group list editor
 *                     (`config-array-input-{key}-{index}`)
 *
 * `definition_id` lives in MODEL_SELECTION so the screen can set the active
 * model on the real form control. STRATEGY is rendered un-collapsed by default
 * (only Advanced Engine / Sampling / Expert Features start collapsed), so its
 * fields are immediately visible to assert against.
 */
export const trainingSchema = {
    type: 'object',
    title: 'Standard Training',
    properties: {
        // MODEL_SELECTION (hardcoded, always-open section)
        definition_id: {
            type: 'string',
            title: 'Model Definition',
            group: 'MODEL_SELECTION',
            enum: ['flux-dev', 'flux-schnell'],
            enum_labels: { 'flux-dev': 'FLUX.1 Dev', 'flux-schnell': 'FLUX.1 Schnell' },
            default: 'flux-dev',
        },
        // BASE → "General Settings"
        lora_name: {
            type: 'string',
            title: 'LoRA Name',
            group: 'BASE',
            default: 'my_lora',
        },
        // STRATEGY → "Training Dynamics" (open by default)
        max_train_steps: {
            type: 'integer',
            title: 'Max Train Steps',
            group: 'STRATEGY',
            default: 1000,
            min: 1,
            step: 100,
        },
        gradient_checkpointing: {
            type: 'boolean',
            title: 'Gradient Checkpointing',
            group: 'STRATEGY',
            default: false,
        },
        // Generic primitive (string) array — renders the standard list editor
        // (`config-array-input-{key}-{index}`), NOT the special-cased
        // `resolutions` preset grid, so the array assertion is straightforward.
        trigger_words: {
            type: 'array',
            title: 'Trigger Words',
            group: 'STRATEGY',
            items: { type: 'string' },
            default: ['ohwx'],
        },
        // OPTIMIZER → "Optimizer Settings"
        optimizer_type: {
            type: 'string',
            title: 'Optimizer',
            group: 'OPTIMIZER',
            enum: ['AdamW', 'AdamW8bit', 'Prodigy'],
            default: 'AdamW',
        },
    },
    required: ['definition_id'],
};

/**
 * GET /api/models/capabilities/{id} — capability descriptor. `enriched: false`
 * keeps the advanced-vram-card in its "not enriched" state (no extra fetches);
 * an empty `field_visibility` hides nothing, so every schema field renders.
 */
export const modelCapabilities = {
    enriched: false,
    block_topology: [] as unknown[],
    lora_targetable_modules: [] as string[],
    trainable_layers: [] as string[],
    archetype: 'diffusion',
    capabilities: {
        has_vae: true,
        has_external_te: true,
        latent_cache: true,
        te_cache: true,
        supports_train_te: false,
        supports_te_quantization: false,
        supports_block_swap: false,
    },
    field_visibility: {} as Record<string, { supported: boolean }>,
    defaults: {} as Record<string, unknown>,
};

/** GET /api/models/settings — global model settings (browse default path). */
export const modelSettings = {
    global_offline_mode: false,
    default_model_path: '/models',
};

/**
 * GET /api/models/definitions/{id}/source — per-definition source override.
 * `hf_hub` (no override) keeps the source badge in its default state.
 */
export const modelSource = {
    source_type: 'hf_hub',
    local_path: null as string | null,
    skip_update: false,
};

/** GET /api/templates/training — empty list, so the selector synthesizes its
 *  built-in "Default" entry (no real template to auto-apply). */
export const trainingTemplates: unknown[] = [];

/**
 * POST /api/jobs/estimate — full calibrated training estimate. The VRAM report
 * feeds both vram-budget-card and the shell rail; `fits: true` shows the FITS
 * chip. peak 18.0 GB / available 24.0 GB are the asserted fixture numbers.
 */
export const trainingEstimate = {
    definition_id: 'flux-dev',
    stats_available: false,
    samples: 0,
    updated_at: null as number | null,
    wall_time: { display: '12m 30s', samples: 0, calibrated: false, seconds: 750 },
    output_size: { display: '320 MB', samples: 0, calibrated: false, bytes: 335_544_320 },
    throughput: { display: '1.33 it/s', samples: 0, calibrated: false, steps_per_sec: 1.33 },
    disk_footprint: { display: '1.2 GB', samples: 0, calibrated: false, bytes: 1_288_490_188 },
    vram: {
        model_weights_mb: 11_500,
        lora_adapters_mb: 200,
        optimizer_states_mb: 800,
        gradients_mb: 400,
        activations_mb: 3_500,
        overhead_mb: 1_024,
        caching_peak_mb: 6_000,
        training_peak_mb: 18_432,
        peak_mb: 18_432,
        available_mb: 24_576,
        total_mb: 24_576,
        used_mb: 0,
        fits: true,
        warnings: [] as string[],
        calibrated: false,
        calibrated_components: [] as string[],
    },
};

/** POST /api/jobs — queue ack returned when the submit path is exercised. */
export const queuedJob = { id: 'job-e2e-1', status: 'pending' };

/* ════════════════════════════════════════════════════════════════════════
 * Dataset workspace + caption/masking settings (Flow C) fixtures.
 *
 * Flow C opens the fullscreen dataset workspace over a dataset card and drives
 * the shared caption/masking settings UIs hosted by the mass-caption /
 * mass-mask modals. Shapes mirror the REAL interfaces the path consumes:
 *   - DatasetPair        → src/app/services/dataset.ts (GET /datasets/{n}/pairs)
 *   - ProjectPreferences → src/app/services/project.service.ts
 *   - Template           → src/app/services/template.service.ts
 *
 * The open + modal path, in order:
 *   click dataset-card-alpha → overlay.openWorkspace(alpha)
 *     workspace.ensurePairsLoaded → GET /api/datasets/alpha/pairs (refreshDataset)
 *   click ws-mass-caption-btn → mass-caption modal
 *     ngOnInit         → GET /api/datasets/alpha/pairs
 *     caption-settings → GET /api/projects/general/preferences
 *                        GET /api/templates/captioning?model_id=florence-2
 *   click ws-mass-mask-btn → mass-mask modal (Generate tab → masking-settings)
 *     ngOnInit         → GET /api/datasets/alpha/pairs
 *     masking-settings → GET /api/projects/general/preferences
 *                        GET /api/templates/masking?model_id=sam3
 *
 * The model `<select>`s in both settings components are populated from a
 * HARDCODED in-component list (florence-2 / qwen3-vl / … and sam3 / rembg), so
 * `caption-model-select` / `masking-model-select` render regardless of the API.
 * The template `<select>`s and the masking `masking-param-*` rows DO depend on
 * the mocked responses below — a non-empty template list keeps each template
 * select populated, and `selected_mask_model: 'sam3'` keeps the sam3 params
 * (e.g. `masking-param-text_prompt`) rendered.
 * ════════════════════════════════════════════════════════════════════════ */

/**
 * GET /api/datasets/{name}/pairs — image-caption pairs for dataset `alpha`.
 *
 * The first pair (`img001.png`, no overlay) is the original Flow C/D fixture:
 * the workspace lands on it as the edit-mode cursor and both mass-caption /
 * mass-mask modals open against it. Flow D's editor enters EDIT mode on this
 * pair, so it is kept first and unchanged.
 *
 * The remaining three pairs feed the mass-edit modal (Flow E):
 *   - `img002.png` has `metadata.has_overlay === true` → the SINGLE valid
 *     SOURCE candidate (the modal's `sourceCandidates` filters on `has_overlay`),
 *     whose recipe is cloned. It is excluded from the target grid once picked.
 *   - `img003.png` is a plain TARGET candidate (no overlay).
 *   - `img004.png` is a TARGET candidate that ALSO has an overlay, so the
 *     `mass-edit-override-badge` "OVR" renders on its tile (asserted in Flow E).
 */
export const datasetPairs = [
    {
        stem: 'img001',
        media_file: 'img001.png',
        media_type: 'image' as const,
        caption_file: 'img001.txt',
        size_bytes: 1_200_000,
        caption_content: 'a portrait photo',
        masked_caption_content: null as string | null,
        metadata: {
            enabled: true,
            has_mask: true,
            has_masked: false,
            has_masked_caption: false,
            has_overlay: false,
            width: 1024,
            height: 1024,
            aspect_ratio: 1.0,
            quality_score: 0.31,
        },
    },
    {
        stem: 'img002',
        media_file: 'img002.png',
        media_type: 'image' as const,
        caption_file: 'img002.txt',
        size_bytes: 1_300_000,
        caption_content: 'a graded portrait',
        masked_caption_content: null as string | null,
        metadata: {
            enabled: true,
            has_mask: true,
            has_masked: false,
            has_masked_caption: false,
            has_overlay: true,
            width: 1024,
            height: 1024,
            aspect_ratio: 1.0,
            quality_score: 0.42,
        },
    },
    {
        stem: 'img003',
        media_file: 'img003.png',
        media_type: 'image' as const,
        caption_file: 'img003.txt',
        size_bytes: 1_100_000,
        caption_content: 'an outdoor shot',
        masked_caption_content: null as string | null,
        metadata: {
            enabled: true,
            has_mask: false,
            has_masked: false,
            has_masked_caption: false,
            has_overlay: false,
            width: 1024,
            height: 768,
            aspect_ratio: 1.333,
            quality_score: 0.28,
        },
    },
    {
        stem: 'img004',
        media_file: 'img004.png',
        media_type: 'image' as const,
        caption_file: 'img004.txt',
        size_bytes: 1_400_000,
        caption_content: 'a studio shot',
        masked_caption_content: null as string | null,
        metadata: {
            enabled: true,
            has_mask: true,
            has_masked: false,
            has_masked_caption: false,
            has_overlay: true,
            width: 1024,
            height: 1024,
            aspect_ratio: 1.0,
            quality_score: 0.36,
        },
    },
];

/**
 * GET /api/projects/general/preferences — the Global-scope preference row both
 * settings components read on init (projects fixture is empty → scope stays
 * Global → effectiveProjectId() is null → backend keys it as 'general').
 * `selected_caption_model` / `selected_mask_model` are in-list ids so the
 * components honor them; the `active_*_template` ids match the templates below
 * so each template select auto-selects a real entry.
 */
export const projectPreferences = {
    id: 'prefs-general',
    project_id: null as string | null,
    selected_caption_model: 'florence-2',
    active_caption_template: 'cap-tpl-1',
    qwen3_variant: '4B-Instruct',
    selected_mask_model: 'sam3',
    active_mask_template: 'mask-tpl-1',
    training_selections: {} as Record<string, unknown>,
};

const TPL_NOW = Math.floor(Date.now() / 1000);

/**
 * GET /api/templates/captioning?model_id=florence-2 — one default template so
 * the `caption-template-select` renders a populated option list and
 * `applyActiveTemplate` seeds the system prompt.
 */
export const captionTemplates = [
    {
        id: 'cap-tpl-1',
        name: 'Default Caption',
        project_id: null as string | null,
        config: {} as Record<string, unknown>,
        created_at: TPL_NOW,
        updated_at: TPL_NOW,
        used_count: 0,
        is_default: true,
        readonly: false,
        model_id: 'florence-2',
        system_prompt: 'Describe this image in detail.',
        wildcard: '',
    },
];

/**
 * GET /api/templates/masking?model_id=sam3 — one default template so the
 * `masking-template-select` renders a populated option list. The sam3 model's
 * params (text_prompt / multimask_output / …) render from the component's
 * hardcoded config once `activeModelConfig()` resolves to sam3.
 */
export const maskingTemplates = [
    {
        id: 'mask-tpl-1',
        name: 'Default Mask',
        project_id: null as string | null,
        config: {} as Record<string, unknown>,
        created_at: TPL_NOW,
        updated_at: TPL_NOW,
        used_count: 0,
        is_default: true,
        readonly: false,
        model_id: 'sam3',
    },
];

/* ════════════════════════════════════════════════════════════════════════
 * Dataset-viewer editors (Flow D / E5) fixtures.
 *
 * Flow D opens the fullscreen workspace, switches to EDIT mode, and drives the
 * real curves / HSL / histogram editors. The live preview is rendered ENTIRELY
 * client-side by PreviewPipeline (it loads the image pixels via a crossOrigin
 * <img>, runs the recipe in a <canvas>, and computes the histogram in-browser);
 * adjusting a slider does NOT hit the network. The only editor → backend round
 * trip is SAVE, which POSTs `/api/datasets/{name}/render-pipeline` and consumes
 * `RenderPipelineResponse` (src/app/services/dataset.ts):
 *   { status, file, overlay, dimensions:[w,h], hash }
 *
 * The Save success toast reads `dimensions`, so a real [w,h] keeps the path
 * happy. `overlay` points at a media path the `/media/**` handler already
 * serves as a PNG.
 * ════════════════════════════════════════════════════════════════════════ */
export const renderPipelineResponse = {
    status: 'ok',
    file: 'img001.png',
    overlay: 'overlays/img001.png',
    dimensions: [64, 64] as number[],
    hash: 'e2eovhash',
};

/* ════════════════════════════════════════════════════════════════════════
 * Mass-apply overlays (Flow E) fixtures.
 *
 * Flow E opens the mass-edit modal ("Apply one image's overlay recipe to
 * many"), picks the SOURCE image whose overlay recipe to clone, selects
 * TARGET images, then submits. Two backend round-trips drive the assertions:
 *   - GET  /api/datasets/{name}/overlay-recipe/{imagePath}
 *       → OverlayRecipeResponse (src/app/services/dataset.ts):
 *         { image_path, recipe: Record<string, unknown> }
 *       The modal reads `res.recipe.operations` (a list of
 *       { type, params?, enabled? }) and renders one "RECIPE SUMMARY" row per
 *       op. A non-empty `operations` list is REQUIRED — an empty list makes the
 *       modal clear the recipe + toast "No pipeline operations found", leaving
 *       Apply disabled. Three ops keep the summary populated.
 *   - POST /api/datasets/{name}/render-pipeline/batch
 *       → { task_id } (the modal then tracks progress via TaskStore/WS). The
 *         e2e flow asserts the POST fires; it does NOT drive task completion.
 * ════════════════════════════════════════════════════════════════════════ */

/** GET /api/datasets/{name}/overlay-recipe/{imagePath} — the source image's
 *  cloned overlay recipe. `recipe.operations` is what the modal consumes. */
export const overlayRecipe = {
    image_path: 'img002.png',
    recipe: {
        operations: [
            { type: 'curves', enabled: true, params: { channel: 'master', points: 5 } },
            { type: 'hsl_selective', enabled: true, params: { range: 'reds', sat: 40 } },
            { type: 'white_balance', enabled: true, params: { temp: 12, tint: -4 } },
        ],
    },
};

/** POST /api/datasets/{name}/render-pipeline/batch — the batch-apply task ack. */
export const batchRenderResponse = { task_id: 'task-e2e' };

/** GET /api/llm-refine/models — topbar's app-init LLM-endpoint availability probe. */
export const llmRefineModels = { curated: [], installed: [], available: false };

/** GET /api/captions/api-providers — dataset-caption-settings loads this list on
 *  init. Empty = no API captioning providers configured (idle default). */
export const apiCaptionProviders: unknown[] = [];

/** GET /api/system/update/status — SystemUpdateService's constructor refreshStatus(). */
export const systemUpdateStatus = {
    state: 'idle',
    available: false,
    branch: 'main',
    commit: 'e2e0000',
    dirty: false,
    is_repo: true,
    behind: 0,
    active: 0,
    error: null,
};

/* ════════════════════════════════════════════════════════════════════════
 * Jobs screen (Flow F) fixtures.
 *
 * `/jobs` mounts `training-job-queue` (left pane) + `JobsScreen`'s own detail
 * pane (center) + `app-system-monitor` (right pane, WS-only — no HTTP). On
 * boot, `training-job-queue.ngOnInit()` fires:
 *   GET /api/jobs/settings/auto-queue   → jobAutoQueue
 *   GET /api/jobs/settings/auto-resume  → jobAutoResume
 *   GET /api/jobs                       → `jobs` (declared above; 1 RUNNING
 *                                          + 1 PENDING row)
 *   GET /api/jobs/history               → jobHistory (empty — no archived
 *                                          rows, so the archive's lazy
 *                                          resumable-checkpoint fetch never
 *                                          fires; out of scope for this smoke
 *                                          flow)
 * `JobsScreen`'s constructor also calls `jobStore.loadAll()` (same GET
 * /api/jobs) and `app-system-monitor.ngOnInit()` calls `jobStore.loadHistory()`
 * (same GET /api/jobs/history) — both hit routes already listed above.
 *
 * Selecting ANY row (`job-item-{id}` click → `viewState.select(id)`) then
 * triggers JobsScreen's per-selection effects, which always fetch (regardless
 * of job status):
 *   GET /api/jobs/{id}/samples          → [] (jobSamplesEmpty)
 *   GET /api/jobs/{id}/checkpoints      → [] (jobCheckpointsEmpty)
 * The fixture jobs are deliberately non-archived/non-sampling (no
 * `sample_every_n_steps`) so the archived-only (getJobReplay/getJobLogs) and
 * sampling-only (getSamplingStatus/getSamplingCadence) effects never fire —
 * kept out of scope for this smoke-level flow.
 *
 * The auto-resume toggle (`auto-resume-toggle` checkbox) PUTs
 * /api/jobs/settings/auto-resume on flip; the handler echoes the posted
 * `enabled` flag back as `auto_resume` so the optimistic UI flip is confirmed
 * (not rolled back).
 */

/** GET /api/jobs/history — no archived rows; keeps the RECENT/ARCHIVE queue
 *  section and its lazy checkpoint-fetch effect out of scope for this flow. */
export const jobHistory: unknown[] = [];

/** GET /api/jobs/settings/auto-queue — off by default (matches component default). */
export const jobAutoQueue = { auto_queue: false };

/** GET /api/jobs/settings/auto-resume — on by default; the e2e flow flips it off. */
export const jobAutoResume = { auto_resume: true };

/** GET /api/jobs/{id}/samples — no samples for either seeded job. */
export const jobSamplesEmpty: unknown[] = [];

/** GET /api/jobs/{id}/checkpoints — no checkpoints for either seeded job. */
export const jobCheckpointsEmpty: unknown[] = [];

/**
 * GET /api/system/gpu/loaded — nothing resident (app-init probe on every page:
 * GpuResidencyStore.refresh()). Shape mirrors `GpuLoadedResponse` in
 * `backend/app/api/system_routes.py`: the three GPU-plugin services in the
 * backend's `_GPU_SERVICES` order, all idle, `any_loaded` precomputed false.
 */
export const gpuLoadedIdle = {
    any_loaded: false,
    services: [
        { service: 'caption', label: 'Captioning', loaded: false, model: null },
        { service: 'masking', label: 'Masking', loaded: false, model: null },
        { service: 'scoring', label: 'Scoring', loaded: false, model: null },
    ],
};

/**
 * GET /api/datasets/thumbnails/legacy — no dataset still holds flat-layout
 * thumbnails (datasets screen, LANE-40 banner). Shape mirrors
 * `ThumbnailLegacySurveyResponse` in `backend/app/api/dataset/thumbnail_routes.py`;
 * `total_files === 0` is the one fact the banner's positive-only `@if` gates on.
 */
export const legacyThumbnailsNone = {
    datasets: [] as unknown[],
    dataset_count: 0,
    total_files: 0,
    total_bytes: 0,
};

/**
 * GET /api/jobs/history/{id}/replay — the jobs screen's detail pane replays the
 * selected run's loss curve. Shape mirrors `JobReplayResponse` in
 * `backend/app/api/training/history_routes.py`; `source:'none'` + empty loss
 * is what the backend returns for a run with neither a disk log nor metrics.
 */
export const jobReplayNone = {
    available: false,
    source: 'none',
    output_dir: null,
    loss: [] as unknown[],
};

/**
 * GET /api/jobs/history/{id}/adaptive — the durable adaptive-targeting
 * timeline. Always 200: a run that never used the feature returns the EMPTY
 * shape (`JobAdaptiveHistoryResponse`, same file), and the UI keys the section
 * on `events.length`.
 */
export const jobAdaptiveNone = {
    events: [] as unknown[],
    modules: [] as string[],
    heat: {} as Record<string, number | null>,
};

/* ════════════════════════════════════════════════════════════════════════
 * Distillation handling (distillation-handling.spec.ts) fixtures.
 *
 * Generated from the backend, never edited by hand: resolve_capabilities for
 * MiniMax-H3-Image (distilled: a training adapter and contrastive guidance)
 * and SDXL (not distilled), and the training schema's own properties for the
 * fields the spec checks. backend/tests/engine/test_distillation_config.py
 * fails when they drift, and its message carries the block to paste here.
 * The spec serves these through its own routes (the boot table's flux
 * fixtures above stay as they are).
 * ════════════════════════════════════════════════════════════════════════ */
// ── BEGIN GENERATED: distillation handling (backend/tests/engine/test_distillation_config.py) ──
export const distillationModels = [
    {
        "id": "minimax-h3-image-t2i",
        "name": "MiniMax-H3-Image (text -> image)",
        "family": "minimax_h3_image",
        "architecture_params": {}
    },
    {
        "id": "sdxl_base_1.0",
        "name": "Stable Diffusion XL Base",
        "family": "sdxl",
        "architecture_params": {}
    }
];
export const distillationCapabilities = {
    "minimax-h3-image-t2i": {
        "archetype": "latent_diffusion",
        "capabilities": {
            "has_vae": true,
            "has_external_te": true,
            "latent_cache": true,
            "te_cache": true,
            "supports_train_te": false,
            "supports_te_quantization": true,
            "supports_block_swap": true,
            "is_video": false,
            "has_audio": false,
            "dual_expert": false,
            "has_image_encoder": false,
            "is_audio_family": false,
            "supports_spatial_resolution": true,
            "supports_dual_sigma_shift": false,
            "supports_cfg_augmentation": false,
            "supports_keyframe_marker": false,
            "supports_min_snr": false,
            "supports_noise_offset": true,
            "supports_flow_timestep_sampling": true,
            "seed_prompt_defaults": true,
            "control_inputs": 0,
            "is_edit": false,
            "supports_augmentation": true,
            "supports_masking_variants": true,
            "supports_distillation_handling": true,
            "distillation": {
                "step": true,
                "guidance": true,
                "handlings": [
                    "adapter",
                    "contrastive_guidance",
                    "both",
                    "none"
                ],
                "default_handling": "both",
                "has_training_adapter": true,
                "guidance_scale_defaults": {
                    "contrastive_guidance": 3.5,
                    "both": 2.5
                },
                "guidance_schedules": [
                    "constant",
                    "sigma"
                ],
                "guidance_schedule_defaults": {
                    "contrastive_guidance": "constant",
                    "both": "sigma"
                },
                "default_guidance_schedule": "sigma"
            }
        },
        "field_visibility": {
            "cache_latents": {
                "supported": true
            },
            "low_vram": {
                "supported": true
            },
            "cache_text_embeddings": {
                "supported": true
            },
            "unload_text_encoder": {
                "supported": true
            },
            "train_text_encoder": {
                "supported": false,
                "reason": "no trainable text encoder for this family"
            },
            "te_quantization": {
                "supported": true
            },
            "te_quantization_backend": {
                "supported": true
            },
            "block_swap_config": {
                "supported": true
            },
            "h_flip": {
                "supported": true
            },
            "v_flip": {
                "supported": true
            },
            "masking_enabled": {
                "supported": true
            },
            "control_resolution": {
                "supported": false,
                "reason": "control resolution only applies to paired edit models"
            },
            "num_frames": {
                "supported": false,
                "reason": "image model — no video frames"
            },
            "target_fps": {
                "supported": false,
                "reason": "image model — no video frames"
            },
            "video_mode": {
                "supported": false,
                "reason": "image model — no video frames"
            },
            "i2v_image_dropout": {
                "supported": false,
                "reason": "image model — no video frames"
            },
            "first_frame_conditioning_probability": {
                "supported": false,
                "reason": "image model — no i2v conditioning"
            },
            "keyframe_marker": {
                "supported": false,
                "reason": "this model has no keyframe embedding"
            },
            "train_audio": {
                "supported": false,
                "reason": "this model has no audio modality"
            },
            "audio_loss_weight": {
                "supported": false,
                "reason": "this model has no audio modality"
            },
            "sigma_shift_video": {
                "supported": false,
                "reason": "this model has a single flow-matching clock — no per-stream sigma shift"
            },
            "sigma_shift_audio": {
                "supported": false,
                "reason": "this model has a single flow-matching clock — no per-stream sigma shift"
            },
            "cfg_augment_scale": {
                "supported": false,
                "reason": "this model does not train on the guidance-rearranged output"
            },
            "expert_mode": {
                "supported": false,
                "reason": "single-transformer model — no expert routing"
            },
            "expert_swap_mode": {
                "supported": false,
                "reason": "single-transformer model — no expert routing"
            },
            "expert_switch_interval": {
                "supported": false,
                "reason": "single-transformer model — no expert routing"
            },
            "temporal_coverage": {
                "supported": false,
                "reason": "image model — no temporal windows"
            },
            "window_overlap": {
                "supported": false,
                "reason": "image model — no temporal windows"
            },
            "max_windows": {
                "supported": false,
                "reason": "image model — no temporal windows"
            },
            "frame_stride": {
                "supported": false,
                "reason": "image model — no video frames to stride"
            },
            "sliding_max_clip_seconds": {
                "supported": false,
                "reason": "image model — no temporal windows"
            },
            "still_resolutions": {
                "supported": false,
                "reason": "image model — no still/video split"
            },
            "resolutions": {
                "supported": true
            },
            "bucketing_mode": {
                "supported": true
            },
            "duration_s": {
                "supported": false,
                "reason": "this model has no audio-generation modality"
            },
            "genre_ratio": {
                "supported": false,
                "reason": "this model has no audio-generation modality"
            },
            "min_snr_gamma": {
                "supported": false,
                "reason": "Min-SNR loss weighting applies to SDXL only"
            },
            "noise_offset": {
                "supported": true
            },
            "timestep_sampling": {
                "supported": true
            },
            "fixed_sigmas": {
                "supported": true
            },
            "logit_normal_mu": {
                "supported": true
            },
            "logit_normal_sigma": {
                "supported": true
            },
            "model_shift_std": {
                "supported": true
            },
            "timestep_uniform_prob": {
                "supported": true
            },
            "mode_scale": {
                "supported": true
            },
            "flux_shift_base": {
                "supported": true
            },
            "flux_shift_max": {
                "supported": true
            },
            "radc_start": {
                "supported": true
            },
            "radc_end": {
                "supported": true
            },
            "radc_width": {
                "supported": true
            },
            "radc_res_influence": {
                "supported": true
            },
            "distillation_handling": {
                "supported": true
            },
            "distillation_guidance_scale": {
                "supported": true
            },
            "distillation_guidance_schedule": {
                "supported": true
            }
        },
        "defaults": {
            "resolution": 1024,
            "scheduler": "flow_match",
            "learning_rate": 0.0001,
            "mixed_precision": "bf16",
            "num_inference_steps": 6,
            "guidance_scale": 1.0,
            "is_distilled": true,
            "timestep_sampling": "fixed_sigmas",
            "fixed_sigmas": [
                1.0,
                0.9675236940383911,
                0.8875485062599182,
                0.75,
                0.5327739119529724,
                0.23200887441635132
            ],
            "distillation_handling": "both",
            "distillation_guidance_schedule": "sigma",
            "distillation_guidance_scale": 2.5
        }
    },
    "sdxl_base_1.0": {
        "archetype": "latent_diffusion",
        "capabilities": {
            "has_vae": true,
            "has_external_te": true,
            "latent_cache": true,
            "te_cache": true,
            "supports_train_te": true,
            "supports_te_quantization": true,
            "supports_block_swap": true,
            "is_video": false,
            "has_audio": false,
            "dual_expert": false,
            "has_image_encoder": false,
            "is_audio_family": false,
            "supports_spatial_resolution": true,
            "supports_dual_sigma_shift": false,
            "supports_cfg_augmentation": false,
            "supports_keyframe_marker": false,
            "supports_min_snr": true,
            "supports_noise_offset": true,
            "supports_flow_timestep_sampling": false,
            "seed_prompt_defaults": false,
            "control_inputs": 0,
            "is_edit": false,
            "supports_augmentation": true,
            "supports_masking_variants": true,
            "supports_distillation_handling": false
        },
        "field_visibility": {
            "cache_latents": {
                "supported": true
            },
            "low_vram": {
                "supported": true
            },
            "cache_text_embeddings": {
                "supported": true
            },
            "unload_text_encoder": {
                "supported": true
            },
            "train_text_encoder": {
                "supported": true
            },
            "te_quantization": {
                "supported": true
            },
            "te_quantization_backend": {
                "supported": true
            },
            "block_swap_config": {
                "supported": true
            },
            "h_flip": {
                "supported": true
            },
            "v_flip": {
                "supported": true
            },
            "masking_enabled": {
                "supported": true
            },
            "control_resolution": {
                "supported": false,
                "reason": "control resolution only applies to paired edit models"
            },
            "num_frames": {
                "supported": false,
                "reason": "image model — no video frames"
            },
            "target_fps": {
                "supported": false,
                "reason": "image model — no video frames"
            },
            "video_mode": {
                "supported": false,
                "reason": "image model — no video frames"
            },
            "i2v_image_dropout": {
                "supported": false,
                "reason": "image model — no video frames"
            },
            "first_frame_conditioning_probability": {
                "supported": false,
                "reason": "image model — no i2v conditioning"
            },
            "keyframe_marker": {
                "supported": false,
                "reason": "this model has no keyframe embedding"
            },
            "train_audio": {
                "supported": false,
                "reason": "this model has no audio modality"
            },
            "audio_loss_weight": {
                "supported": false,
                "reason": "this model has no audio modality"
            },
            "sigma_shift_video": {
                "supported": false,
                "reason": "this model has a single flow-matching clock — no per-stream sigma shift"
            },
            "sigma_shift_audio": {
                "supported": false,
                "reason": "this model has a single flow-matching clock — no per-stream sigma shift"
            },
            "cfg_augment_scale": {
                "supported": false,
                "reason": "this model does not train on the guidance-rearranged output"
            },
            "expert_mode": {
                "supported": false,
                "reason": "single-transformer model — no expert routing"
            },
            "expert_swap_mode": {
                "supported": false,
                "reason": "single-transformer model — no expert routing"
            },
            "expert_switch_interval": {
                "supported": false,
                "reason": "single-transformer model — no expert routing"
            },
            "temporal_coverage": {
                "supported": false,
                "reason": "image model — no temporal windows"
            },
            "window_overlap": {
                "supported": false,
                "reason": "image model — no temporal windows"
            },
            "max_windows": {
                "supported": false,
                "reason": "image model — no temporal windows"
            },
            "frame_stride": {
                "supported": false,
                "reason": "image model — no video frames to stride"
            },
            "sliding_max_clip_seconds": {
                "supported": false,
                "reason": "image model — no temporal windows"
            },
            "still_resolutions": {
                "supported": false,
                "reason": "image model — no still/video split"
            },
            "resolutions": {
                "supported": true
            },
            "bucketing_mode": {
                "supported": true
            },
            "duration_s": {
                "supported": false,
                "reason": "this model has no audio-generation modality"
            },
            "genre_ratio": {
                "supported": false,
                "reason": "this model has no audio-generation modality"
            },
            "min_snr_gamma": {
                "supported": true
            },
            "noise_offset": {
                "supported": true
            },
            "timestep_sampling": {
                "supported": false,
                "reason": "this model draws a discrete uniform timestep — no flow-matching timestep sampling"
            },
            "fixed_sigmas": {
                "supported": false,
                "reason": "this model draws a discrete uniform timestep — no flow-matching timestep sampling"
            },
            "logit_normal_mu": {
                "supported": false,
                "reason": "this model draws a discrete uniform timestep — no flow-matching timestep sampling"
            },
            "logit_normal_sigma": {
                "supported": false,
                "reason": "this model draws a discrete uniform timestep — no flow-matching timestep sampling"
            },
            "model_shift_std": {
                "supported": false,
                "reason": "this model draws a discrete uniform timestep — no flow-matching timestep sampling"
            },
            "timestep_uniform_prob": {
                "supported": false,
                "reason": "this model draws a discrete uniform timestep — no flow-matching timestep sampling"
            },
            "mode_scale": {
                "supported": false,
                "reason": "this model draws a discrete uniform timestep — no flow-matching timestep sampling"
            },
            "flux_shift_base": {
                "supported": false,
                "reason": "this model draws a discrete uniform timestep — no flow-matching timestep sampling"
            },
            "flux_shift_max": {
                "supported": false,
                "reason": "this model draws a discrete uniform timestep — no flow-matching timestep sampling"
            },
            "radc_start": {
                "supported": false,
                "reason": "this model draws a discrete uniform timestep — no flow-matching timestep sampling"
            },
            "radc_end": {
                "supported": false,
                "reason": "this model draws a discrete uniform timestep — no flow-matching timestep sampling"
            },
            "radc_width": {
                "supported": false,
                "reason": "this model draws a discrete uniform timestep — no flow-matching timestep sampling"
            },
            "radc_res_influence": {
                "supported": false,
                "reason": "this model draws a discrete uniform timestep — no flow-matching timestep sampling"
            },
            "distillation_handling": {
                "supported": false,
                "reason": "this model is not distilled — it trains without a distillation handling"
            },
            "distillation_guidance_scale": {
                "supported": false,
                "reason": "this model is not distilled — it trains without a distillation handling"
            },
            "distillation_guidance_schedule": {
                "supported": false,
                "reason": "this model is not distilled — it trains without a distillation handling"
            }
        },
        "defaults": {
            "resolution": 1024,
            "scheduler": "euler_a",
            "learning_rate": 0.0001,
            "num_inference_steps": 50,
            "guidance_scale": 5.0
        }
    }
};
export const distillationSchemaProperties = {
    "timestep_sampling": {
        "default": "logit_normal",
        "description": "Timestep sampling strategy for training (HiDream-O1 runs flux_shift/model_shift as uniform and ignores radc_res_influence; SDXL hides it)",
        "enum": [
            "logit_normal",
            "uniform",
            "sigmoid",
            "cosmap",
            "mode",
            "flux_shift",
            "radc",
            "model_shift",
            "fixed_sigmas"
        ],
        "group": "STRATEGY",
        "title": "Timestep Sampling",
        "type": "string"
    },
    "fixed_sigmas": {
        "anyOf": [
            {
                "items": {
                    "type": "number"
                },
                "type": "array"
            },
            {
                "type": "null"
            }
        ],
        "default": null,
        "depends_on": "timestep_sampling:fixed_sigmas",
        "description": "The sigmas fixed_sigmas draws from, uniformly (each in (0, 1]; a model distilled to a fixed schedule trains on its levels)",
        "group": "STRATEGY",
        "title": "Fixed Sigmas"
    },
    "distillation_handling": {
        "anyOf": [
            {
                "enum": [
                    "adapter",
                    "contrastive_guidance",
                    "both",
                    "none"
                ],
                "type": "string"
            },
            {
                "type": "null"
            }
        ],
        "default": null,
        "description": "How training compensates for a distilled model: a frozen training adapter, contrastive guidance, both, or none (empty = the model's default)",
        "group": "STRATEGY",
        "title": "Distillation handling"
    },
    "distillation_guidance_scale": {
        "anyOf": [
            {
                "type": "number"
            },
            {
                "type": "null"
            }
        ],
        "default": null,
        "depends_on": "distillation_handling:contrastive_guidance,both",
        "description": "Contrastive guidance: the target is u + s*(v - u), u the blank-prompt prediction; 1 trains as without guidance (empty = the model's default)",
        "group": "STRATEGY",
        "max": 10.0,
        "min": 1.0,
        "step": 0.1,
        "title": "Guidance target"
    },
    "distillation_guidance_schedule": {
        "anyOf": [
            {
                "enum": [
                    "constant",
                    "sigma"
                ],
                "type": "string"
            },
            {
                "type": "null"
            }
        ],
        "default": null,
        "depends_on": "distillation_handling:contrastive_guidance,both",
        "description": "constant: s at every noise level; sigma: 1 + (s - 1) * sigma, weaker toward clean (empty = constant)",
        "group": "STRATEGY",
        "title": "Guidance schedule"
    }
};
// ── END GENERATED: distillation handling ──
