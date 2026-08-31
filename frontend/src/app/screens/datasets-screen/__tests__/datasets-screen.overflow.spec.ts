import type { Mock } from 'vitest';
import { TestBed } from '@angular/core/testing';
import { signal, WritableSignal } from '@angular/core';
import { of } from 'rxjs';
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
 * D4 — responsive card-action overflow.
 *
 * The card action bar keeps its primary actions inline and collapses the
 * secondary ones into a "⋯" overflow popover at narrow card widths (the
 * inline-vs-collapsed swap is a pure-CSS container query; only the popover's
 * open/close state is TS logic). The overflow menu is keyed by dataset name
 * (mirroring the per-card project picker) so it survives grid re-renders, and
 * opening it must not open the card (stopPropagation).
 *
 * Component is instantiated directly (no render) mirroring the sibling specs.
 */
describe('DatasetsScreen — card-action overflow menu', () => {
    let projectId: WritableSignal<string | null>;

    const DS1 = { id: 'd1', name: 'ds-1' };
    const DS2 = { id: 'd2', name: 'ds-2' };
    const ev = () => ({ stopPropagation: vi.fn() }) as unknown as MouseEvent;

    interface Overflow {
        overflowOpenFor: WritableSignal<string>;
        toggleOverflow(name: string, e: MouseEvent): void;
        closeOverflow(): void;
        onDocumentClick(): void;
        onMenuKeydown(e: KeyboardEvent, which: string, trigger: HTMLElement): void;
    }

    function make(): DatasetsScreen & Overflow {
        return TestBed.runInInjectionContext(() => new DatasetsScreen()) as DatasetsScreen & Overflow;
    }

    beforeEach(() => {
        projectId = signal<string | null>(null);

        TestBed.configureTestingModule({
            providers: [
                {
                    provide: DatasetStore,
                    useValue: {
                        loadAll: vi.fn().mockResolvedValue(undefined),
                        loading: signal(false),
                        entities: signal([DS1, DS2]),
                    },
                },
                {
                    provide: DatasetService,
                    useValue: { getCacheStats: () => of(null), getLegacyThumbnailSurvey: () => of({ datasets: [], dataset_count: 0, total_files: 0, total_bytes: 0 }), getMpxDistribution: () => of(null) },
                },
                {
                    provide: ProjectService,
                    useValue: {
                        getProjectDatasets: () => of([]),
                        allProjects: signal([{ id: 'p1', name: 'Proj One' }]),
                    },
                },
                { provide: RuntimeConfigService, useValue: { apiUrl: '/api', mediaBaseUrl: '/media' } },
                { provide: ToastService, useValue: { success: () => {}, error: () => {} } },
                { provide: ScopeStore, useValue: { projectId } },
                { provide: OverlayStore, useValue: { workspace: signal(null), openModal: () => {}, openWorkspace: () => {} } },
                { provide: SearchStore, useValue: { query: signal(''), fields: signal(new Set<string>()) } },
            ],
        });
    });

    it('starts with no overflow menu open', () => {
        const c = make();
        expect(c.overflowOpenFor()).toBe('');
    });

    it('toggleOverflow opens the menu for a card and stops propagation (so the card does not open)', () => {
        const c = make();
        const e = ev();
        c.toggleOverflow('ds-1', e);
        expect(c.overflowOpenFor()).toBe('ds-1');
        expect(e.stopPropagation as Mock).toHaveBeenCalled();
    });

    it('toggleOverflow on the same card closes it', () => {
        const c = make();
        c.toggleOverflow('ds-1', ev());
        c.toggleOverflow('ds-1', ev());
        expect(c.overflowOpenFor()).toBe('');
    });

    it('opening a second card switches which overflow menu is open (only one at a time)', () => {
        const c = make();
        c.toggleOverflow('ds-1', ev());
        c.toggleOverflow('ds-2', ev());
        expect(c.overflowOpenFor()).toBe('ds-2');
    });

    it('closeOverflow resets the open state', () => {
        const c = make();
        c.toggleOverflow('ds-1', ev());
        c.closeOverflow();
        expect(c.overflowOpenFor()).toBe('');
    });

    it('a document click dismisses an open overflow menu', () => {
        const c = make();
        c.toggleOverflow('ds-1', ev());
        c.onDocumentClick();
        expect(c.overflowOpenFor()).toBe('');
    });

    it('Escape inside the overflow menu closes it and restores focus to the trigger', () => {
        const c = make();
        c.toggleOverflow('ds-1', ev());
        const trigger = { focus: vi.fn() } as unknown as HTMLElement;
        const kev = {
            key: 'Escape',
            preventDefault: vi.fn(),
            stopPropagation: vi.fn(),
            currentTarget: document.createElement('div'),
        } as unknown as KeyboardEvent;
        c.onMenuKeydown(kev, 'overflow', trigger);
        expect(c.overflowOpenFor()).toBe('');
        expect(trigger.focus as Mock).toHaveBeenCalled();
    });
});
