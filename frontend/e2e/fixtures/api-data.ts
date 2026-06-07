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

/** GET / (sidebar app-version probe). */
export const version = { version: 'e2e' };

/** GET /api/projects — shell loads this on boot. Empty list keeps the scope
 *  switcher in Global, which is what the boot smoke exercises. */
export const projects: unknown[] = [];

/** GET /api/jobs — not fetched on the datasets boot path, but returned as a
 *  benign empty list if anything probes it. */
export const jobs: unknown[] = [];

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
 * GET /api/datasets/{name}/pairs — a single image-caption pair. The workspace
 * (via DatasetSyncService.refreshDataset) and both mass-* modals fetch this on
 * open; one ready pair is enough to render the filmstrip + let the modals open.
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
