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
