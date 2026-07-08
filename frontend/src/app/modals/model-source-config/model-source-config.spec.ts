import { describe, it, expect, vi, beforeEach } from 'vitest';
import { TestBed, type ComponentFixture } from '@angular/core/testing';
import { of } from 'rxjs';
import { ModelSourceConfigComponent, type ModelSourceConfigData } from './model-source-config';
import { OverlayStore } from '../../state/overlay.store';
import { ModelService, type ModelSourceOverride } from '../../services/model.service';
import { ToastService } from '../../services/toast';
import { RegistryStore } from '../../state/registry.store';

function setup(over: Partial<ModelSourceConfigData> = {}): {
    fixture: ComponentFixture<ModelSourceConfigComponent>;
    comp: any;
    onSaved: ReturnType<typeof vi.fn>;
    overlay: OverlayStore;
    registry: { setOverride: ReturnType<typeof vi.fn>; clearOverride: ReturnType<typeof vi.fn> };
} {
    const onSaved = vi.fn();
    const registry = {
        loadFor: vi.fn().mockResolvedValue(undefined),
        byId: vi.fn().mockReturnValue(() => null),
        setOverride: vi.fn().mockResolvedValue(undefined),
        clearOverride: vi.fn().mockResolvedValue(undefined),
    };
    TestBed.configureTestingModule({
        imports: [ModelSourceConfigComponent],
        providers: [
            { provide: ModelService, useValue: { pickFolder: () => of({ path: '' }), validatePath: () => of(null), getGlobalSettings: () => of({ default_model_path: '' }) } },
            { provide: ToastService, useValue: { success: vi.fn(), error: vi.fn(), warning: vi.fn(), info: vi.fn() } },
            { provide: RegistryStore, useValue: registry },
        ],
    });
    const overlay = TestBed.inject(OverlayStore);
    const data: ModelSourceConfigData = { definitionId: 'flux-dev', definitionName: 'Flux Dev', onSaved, ...over };
    overlay.openModal('model-source-config', data);
    const fixture = TestBed.createComponent(ModelSourceConfigComponent);
    fixture.detectChanges();
    return { fixture, comp: fixture.componentInstance, onSaved, overlay, registry };
}

describe('ModelSourceConfigComponent (registered modal)', () => {
    beforeEach(() => TestBed.resetTestingModule());

    it('reads definition context from the overlay payload', () => {
        const { fixture } = setup();
        const el = fixture.nativeElement as HTMLElement;
        expect(el.querySelector('[data-testid="model-source-type-select"]')).toBeTruthy();
        expect(el.textContent).toContain('Flux Dev');
    });

    it('renders on the shared modal chrome (head/body/foot + house buttons)', () => {
        const { fixture } = setup();
        const el = fixture.nativeElement as HTMLElement;
        // Shared structure
        expect(el.querySelector('.modal-head')).toBeTruthy();
        expect(el.querySelector('.modal-body')).toBeTruthy();
        expect(el.querySelector('.modal-foot')).toBeTruthy();
        // Close control is the shared .icon-btn in the head
        expect(el.querySelector('.modal-head .icon-btn[data-testid="model-source-close-btn"]')).toBeTruthy();
        // House buttons in the footer: Save = .btn.primary, Cancel = .btn.ghost
        const save = el.querySelector('[data-testid="model-source-save-btn"]');
        const cancel = el.querySelector('[data-testid="model-source-cancel-btn"]');
        expect(save?.classList.contains('btn')).toBe(true);
        expect(save?.classList.contains('primary')).toBe(true);
        expect(cancel?.classList.contains('btn')).toBe(true);
        expect(cancel?.classList.contains('ghost')).toBe(true);
        expect(el.querySelector('.modal-foot .btn.primary')).toBeTruthy();
    });

    it('renders Reset as a house danger button when an override exists', () => {
        const { fixture, comp } = setup();
        comp.hasExistingOverride.set(true);
        fixture.detectChanges();
        const reset = fixture.nativeElement.querySelector('[data-testid="model-source-reset-btn"]') as HTMLElement;
        expect(reset).toBeTruthy();
        expect(reset.classList.contains('btn')).toBe(true);
        expect(reset.classList.contains('danger-out')).toBe(true);
    });

    it('save() calls onSaved with the override, persists, and closes', () => {
        const { comp, onSaved, overlay, registry } = setup();
        comp.sourceType.set('local_diffusers');
        comp.localPath.set('D:\\Models\\x');
        comp.save();
        const override: ModelSourceOverride = onSaved.mock.calls[0][0];
        expect(override.source_type).toBe('local_diffusers');
        expect(override.local_path).toBe('D:\\Models\\x');
        expect(registry.setOverride).toHaveBeenCalledWith('flux-dev', override);
        expect(overlay.modalStack().length).toBe(0);
    });

    it('resetToDefault() calls onSaved(null), clears, and closes', () => {
        const { comp, onSaved, overlay, registry } = setup();
        comp.resetToDefault();
        expect(onSaved).toHaveBeenCalledWith(null);
        expect(registry.clearOverride).toHaveBeenCalledWith('flux-dev');
        expect(overlay.modalStack().length).toBe(0);
    });
});
