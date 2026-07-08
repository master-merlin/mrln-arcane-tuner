import { TestBed, type ComponentFixture } from '@angular/core/testing';
import { signal } from '@angular/core';
import { By } from '@angular/platform-browser';
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
 * D10 — keyboard operability.
 *
 * Cards must be keyboard-activatable (role=button + tabindex + Enter/Space →
 * open). The filter-picker popup (role=menu) gains a keyboard model: focus
 * moves into the panel on open, Escape closes + restores focus to the trigger,
 * and arrows rove between role=menuitem entries.
 *
 * These are render tests — the fixture is attached to document.body so
 * `.focus()` actually moves `document.activeElement` under jsdom.
 */
describe('DatasetsScreen — keyboard operability', () => {
    let fixture: ComponentFixture<DatasetsScreen>;
    let openWorkspace: ReturnType<typeof vi.fn>;

    const DS = {
        id: 'd1',
        name: 'ds-1',
        version: '1',
        classifier: 'portrait',
        tags: ['favs'],
        file_count: 1,
        multimedia_count: 1,
        caption_count: 1,
        mask_count: 0,
    };

    function setup(): void {
        openWorkspace = vi.fn();
        TestBed.configureTestingModule({
            imports: [DatasetsScreen],
            providers: [
                {
                    provide: DatasetStore,
                    useValue: {
                        loadAll: () => Promise.resolve(),
                        loading: signal(false),
                        entities: signal([DS]),
                    },
                },
                {
                    provide: DatasetService,
                    useValue: { getCacheStats: () => of(null), getMpxDistribution: () => of(null) },
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
                { provide: ScopeStore, useValue: { projectId: signal<string | null>(null) } },
                {
                    provide: OverlayStore,
                    useValue: { workspace: signal(null), openModal: () => {}, openWorkspace },
                },
                { provide: SearchStore, useValue: { query: signal(''), fields: signal(new Set<string>()) } },
            ],
        });
        fixture = TestBed.createComponent(DatasetsScreen);
        document.body.appendChild(fixture.nativeElement);
        fixture.detectChanges();
    }

    beforeEach(setup);
    afterEach(() => {
        fixture.nativeElement.remove();
    });

    function card(): HTMLElement {
        return fixture.debugElement.query(By.css('[data-testid="dataset-card-ds-1"]')).nativeElement;
    }

    it('renders each card as a keyboard-focusable button', () => {
        const el = card();
        expect(el.getAttribute('role')).toBe('button');
        expect(el.getAttribute('tabindex')).toBe('0');
    });

    it('opens the card on Enter', () => {
        const el = card();
        el.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
        expect(openWorkspace).toHaveBeenCalled();
    });

    it('opens the card on Space', () => {
        const el = card();
        el.dispatchEvent(new KeyboardEvent('keydown', { key: ' ', bubbles: true }));
        expect(openWorkspace).toHaveBeenCalled();
    });

    it('does NOT open the card when Enter is pressed on an inner action button', () => {
        const inner = fixture.debugElement.query(By.css('[data-testid="btn-upload-files"]')).nativeElement as HTMLElement;
        inner.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
        expect(openWorkspace).not.toHaveBeenCalled();
    });

    it('moves focus into the filter picker on open, roves with ArrowDown, and Escape closes + restores focus to the trigger', async () => {
        const trigger = fixture.debugElement.query(By.css('[data-testid="btn-add-filter"]')).nativeElement as HTMLElement;
        trigger.click();
        fixture.detectChanges();
        await settle();
        fixture.detectChanges();

        const panel = fixture.debugElement.query(By.css('[data-testid="filter-picker-panel"]'));
        expect(panel).toBeTruthy();
        const items = panel.nativeElement.querySelectorAll('[role="menuitem"]') as NodeListOf<HTMLElement>;
        expect(items.length).toBeGreaterThan(1);

        // Focus moved to the first item on open.
        expect(document.activeElement).toBe(items[0]);

        // ArrowDown roves to the next item.
        panel.nativeElement.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowDown', bubbles: true }));
        expect(document.activeElement).toBe(items[1]);

        // Escape closes the panel and restores focus to the trigger.
        panel.nativeElement.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
        fixture.detectChanges();
        expect(fixture.debugElement.query(By.css('[data-testid="filter-picker-panel"]'))).toBeNull();
        expect(document.activeElement).toBe(trigger);
    });
});
