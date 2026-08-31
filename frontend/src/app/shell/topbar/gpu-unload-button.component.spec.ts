import { TestBed } from '@angular/core/testing';
import { signal } from '@angular/core';
import { of, throwError } from 'rxjs';
import { GpuUnloadButtonComponent } from './gpu-unload-button.component';
import { GpuResidencyStore, type GpuServiceState } from '../../state/gpu-residency.store';
import { ToastService } from '../../services/toast';

/**
 * Every assertion here reads the RENDERED DOM, never a computed on the
 * component instance. A computed can be right while nothing reaches the
 * template — which is exactly the failure mode that cost a day in this area
 * (a `[ngModel]` select whose options arrived after `writeValue`, rendering
 * blank with `selectedIndex === -1` while every signal read correct).
 */
function setup(opts: {
    anyLoaded: boolean;
    services?: GpuServiceState[];
    unloading?: boolean;
    unloadAll?: () => ReturnType<GpuResidencyStore['unloadAll']>;
}) {
    const unloading = signal(opts.unloading ?? false);
    const store = {
        anyLoaded: signal(opts.anyLoaded),
        services: signal(opts.services ?? []),
        unloading,
        unloadAll: opts.unloadAll ?? (() => of({ any_loaded: false, services: [], unloaded: [], skipped: [] })),
    };
    const toast = { success: vi.fn(), error: vi.fn(), warning: vi.fn(), info: vi.fn() };

    TestBed.configureTestingModule({
        imports: [GpuUnloadButtonComponent],
        providers: [
            { provide: GpuResidencyStore, useValue: store },
            { provide: ToastService, useValue: toast },
        ],
    });
    const fixture = TestBed.createComponent(GpuUnloadButtonComponent);
    fixture.detectChanges();
    const btn = () =>
        fixture.nativeElement.querySelector('[data-testid="gpu-unload-btn"]') as HTMLButtonElement | null;
    return { fixture, btn, store, toast };
}

const CAPTION_ONLY: GpuServiceState[] = [
    { service: 'caption', label: 'Captioning', loaded: true, model: 'florence2:base' },
    { service: 'masking', label: 'Masking', loaded: false, model: null },
    { service: 'scoring', label: 'Scoring', loaded: false, model: null },
];

describe('GpuUnloadButtonComponent', () => {
    it('renders NOTHING when no model is loaded', () => {
        // The negative: a positive-only control that renders anyway is a
        // button offering to free memory that is already free.
        const { btn, fixture } = setup({ anyLoaded: false, services: [] });
        expect(btn()).toBeNull();
        expect(fixture.nativeElement.textContent.trim()).toBe('');
    });

    it('renders the button once something is loaded', () => {
        const { btn } = setup({ anyLoaded: true, services: CAPTION_ONLY });
        expect(btn()).not.toBeNull();
    });

    it('names what is loaded in the tooltip, in the DOM', () => {
        const { btn } = setup({
            anyLoaded: true,
            services: [
                { service: 'caption', label: 'Captioning', loaded: true, model: 'florence2:base' },
                { service: 'scoring', label: 'Scoring', loaded: true, model: 'hpsv2' },
                { service: 'masking', label: 'Masking', loaded: false, model: null },
            ],
        });
        expect(btn()!.getAttribute('title')).toBe('Free GPU memory — Captioning, Scoring loaded');
        expect(btn()!.getAttribute('aria-label')).toBe('Free GPU memory — Captioning, Scoring loaded');
    });

    it('shows a busy, disabled button while the unload is in flight', () => {
        const { btn } = setup({ anyLoaded: true, services: CAPTION_ONLY, unloading: true });
        expect(btn()!.disabled).toBe(true);
        expect(btn()!.getAttribute('aria-busy')).toBe('true');
        expect(btn()!.classList.contains('is-busy')).toBe(true);
        expect(btn()!.getAttribute('title')).toBe('Freeing GPU memory…');
    });

    it('is not busy or disabled at rest', () => {
        const { btn } = setup({ anyLoaded: true, services: CAPTION_ONLY });
        expect(btn()!.disabled).toBe(false);
        expect(btn()!.getAttribute('aria-busy')).toBeNull();
    });

    it('toasts what was freed', () => {
        const { btn, toast } = setup({
            anyLoaded: true,
            services: CAPTION_ONLY,
            unloadAll: () =>
                of({ any_loaded: false, services: [], unloaded: ['caption'], skipped: [] }),
        });
        btn()!.click();
        expect(toast.success).toHaveBeenCalledWith('Freed 1 model(s) — GPU memory released.');
    });

    it('toasts the reason when a busy service was kept', () => {
        const { btn, toast } = setup({
            anyLoaded: true,
            services: CAPTION_ONLY,
            unloadAll: () =>
                of({
                    any_loaded: true,
                    services: [],
                    unloaded: [],
                    skipped: [{ service: 'caption', reason: 'captioning is busy — a batch task is using the model' }],
                }),
        });
        btn()!.click();
        expect(toast.warning).toHaveBeenCalledWith(
            'Nothing freed — captioning is busy — a batch task is using the model.',
        );
        expect(toast.success).not.toHaveBeenCalled();
    });

    it('toasts an error when the request fails', () => {
        const { btn, toast } = setup({
            anyLoaded: true,
            services: CAPTION_ONLY,
            unloadAll: () => throwError(() => new Error('boom')),
        });
        btn()!.click();
        expect(toast.error).toHaveBeenCalledWith('Could not free GPU memory. Check the server logs.');
    });
});
