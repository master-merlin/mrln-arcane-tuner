import { TestBed } from '@angular/core/testing';
import { signal, WritableSignal } from '@angular/core';
import { of } from 'rxjs';
import { settle } from '../../../../testing/async';
import { DatasetsScreen } from '../datasets-screen';
import { DatasetStore } from '../../../state/dataset.store';
import { DatasetService } from '../../../services/dataset';
import { ProjectService } from '../../../services/project.service';
import { RuntimeConfigService } from '../../../services/runtime-config.service';
import { ToastService } from '../../../services/toast';
import { ScopeStore } from '../../../state/scope.store';
import { OverlayStore } from '../../../state/overlay.store';
import { SearchStore } from '../../../state/search.store';

/**
 * HPS presentation (plan item D6).
 *
 * The quality bands already exist as color via `hpsTone()`. This spec pins the
 * presentation refinements layered on top:
 *   - reduced precision (~2 decimals) on both the card pill and the KPI median,
 *   - a tone→word band helper (success→Good / warning→Fair / danger→Low),
 *   - a screen-reader / colorblind-safe pill label that spells the band out as
 *     TEXT (not color alone),
 *   - a lightweight "what is HPS" legend string surfaced near the KPI tile.
 *
 * Instantiated directly (no template render), matching the sibling specs.
 */
describe('DatasetsScreen — HPS presentation (D6)', () => {
    let entities: WritableSignal<Array<Record<string, unknown>>>;
    let loading: WritableSignal<boolean>;
    let query: WritableSignal<string>;

    type HpsHarness = {
        hpsLabel: (d: unknown) => string;
        hpsMedianLabel: () => string;
        hpsTone: (d: unknown) => 'success' | 'warning' | 'danger' | '';
        hpsBandWord: (tone: 'success' | 'warning' | 'danger' | '') => string;
        hpsPillAria: (d: unknown) => string;
        hpsLegend: string;
    };

    function h(comp: DatasetsScreen): HpsHarness {
        return comp as unknown as HpsHarness;
    }

    function make(): DatasetsScreen {
        return TestBed.runInInjectionContext(() => new DatasetsScreen());
    }

    beforeEach(() => {
        entities = signal<Array<Record<string, unknown>>>([]);
        loading = signal(false);
        query = signal('');

        TestBed.configureTestingModule({
            providers: [
                {
                    provide: DatasetStore,
                    useValue: { loadAll: () => Promise.resolve(), entities, loading },
                },
                {
                    provide: DatasetService,
                    useValue: { getCacheStats: () => of(null), getLegacyThumbnailSurvey: () => of({ datasets: [], dataset_count: 0, total_files: 0, total_bytes: 0 }), getMpxDistribution: () => of(null) },
                },
                { provide: ProjectService, useValue: { getProjectDatasets: () => of([]) } },
                { provide: RuntimeConfigService, useValue: { mediaBaseUrl: '' } },
                { provide: ToastService, useValue: { success: () => {}, error: () => {} } },
                { provide: ScopeStore, useValue: { projectId: signal<string | null>(null) } },
                { provide: OverlayStore, useValue: { workspace: signal(null), openModal: () => {} } },
                { provide: SearchStore, useValue: { query, fields: signal(new Set<string>()) } },
            ],
        });
    });

    // ── Precision ──────────────────────────────────────────────────────
    it('renders the card pill value with ~2 decimals (no false precision)', () => {
        const comp = make();
        expect(h(comp).hpsLabel({ median_quality_score: 0.27345 })).toBe('0.27');
    });

    it('still shows the em-dash placeholder when a dataset has no score', () => {
        const comp = make();
        expect(h(comp).hpsLabel({ median_quality_score: null })).toBe('—');
        expect(h(comp).hpsLabel({})).toBe('—');
    });

    it('renders the KPI median with ~2 decimals', async () => {
        entities.set([
            { id: 'a', name: 'a', median_quality_score: 0.2 },
            { id: 'b', name: 'b', median_quality_score: 0.3 },
        ]);
        const comp = make();
        TestBed.tick();
        await settle();
        // median of 0.2 / 0.3 is 0.25 → two decimals, not "0.2500".
        expect(h(comp).hpsMedianLabel()).toBe('0.25');
    });

    // ── Band word helper ────────────────────────────────────────────────
    it('maps tone → Good / Fair / Low', () => {
        const comp = make();
        expect(h(comp).hpsBandWord('success')).toBe('Good');
        expect(h(comp).hpsBandWord('warning')).toBe('Fair');
        expect(h(comp).hpsBandWord('danger')).toBe('Low');
        expect(h(comp).hpsBandWord('')).toBe('');
    });

    it('keeps the band word consistent with hpsTone across the thresholds', () => {
        const comp = make();
        const band = (v: number) => h(comp).hpsBandWord(h(comp).hpsTone({ median_quality_score: v }));
        expect(band(0.28)).toBe('Good'); // ≥ 0.27
        expect(band(0.25)).toBe('Fair'); // ≥ 0.24
        expect(band(0.10)).toBe('Low'); // else
    });

    // ── Text-band a11y wiring ────────────────────────────────────────────
    it('spells the band out as TEXT in the pill aria-label (not color alone)', () => {
        const comp = make();
        const aria = h(comp).hpsPillAria({ median_quality_score: 0.28 });
        expect(aria).toContain('Good');
        expect(aria).toContain('0.28');
        // The whole phrase is screen-reader friendly (mentions HPS).
        expect(aria.toLowerCase()).toContain('hps');
    });

    it('reflects Fair and Low bands in the pill aria-label too', () => {
        const comp = make();
        expect(h(comp).hpsPillAria({ median_quality_score: 0.25 })).toContain('Fair');
        expect(h(comp).hpsPillAria({ median_quality_score: 0.10 })).toContain('Low');
    });

    // ── "What is HPS" legend ─────────────────────────────────────────────
    it('exposes a concise HPS legend naming the Good/Fair/Low thresholds', () => {
        const comp = make();
        const legend = h(comp).hpsLegend;
        expect(typeof legend).toBe('string');
        expect(legend.length).toBeGreaterThan(0);
        expect(legend).toContain('0.27');
        expect(legend).toContain('0.24');
        expect(legend).toContain('Good');
        expect(legend).toContain('Fair');
        expect(legend).toContain('Low');
    });
});
