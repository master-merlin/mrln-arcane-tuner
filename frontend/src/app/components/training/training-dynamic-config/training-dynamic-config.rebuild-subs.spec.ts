import { TestBed } from '@angular/core/testing';
import { of } from 'rxjs';

import { TrainingDynamicConfigComponent } from './training-dynamic-config';
import { DatasetService } from '../../../services/dataset';
import { DatasetStore } from '../../../state/dataset.store';
import { ToastService } from '../../../services/toast';
import { SystemService } from '../../../services/system.service';
import { JobService } from '../../../services/job';
import { ConfigHelpService } from '../../../services/config-help.service';
import { ModelService } from '../../../services/model.service';
import { RegistryStore } from '../../../state/registry.store';
import { ModelCapabilitiesService } from '../../../services/model-capabilities.service';
import type { SchemaNode } from '../schema-node';

/**
 * F-ARCH-8: the schema/model-list rebuild effect (correctly `untracked`) used
 * to `this.form.valueChanges...subscribe(...)` TWICE inside the effect body
 * on every rebuild, relying solely on `takeUntilDestroyed(this.destroyRef)`
 * to ever unsubscribe. Because `buildForm()` swaps in a brand-new `FormGroup`
 * every run, each rebuild's pair of subscriptions stays bound to that run's
 * (now-abandoned) form instance and only clears on component destroy — a
 * stale-closure hazard: anything still holding an older form reference (a
 * template selector, a modal, a captured local) that writes to it will
 * silently re-trigger the VRAM-estimate/auto-save side effects.
 *
 * Fix requirement: exactly ONE live subscription pair regardless of rebuild
 * count — a stale write to an abandoned form generation must NOT fire the
 * handler; only the current generation's edits may.
 *
 * P4d note: the VRAM-estimate side effect moved into VramEstimationService
 * (F-ARCH-11). The single-subscription-pair invariant is unchanged — the
 * `form.valueChanges` subscription still lives in the component's
 * `_formValueSubs` and is disposed per rebuild — so this spec now spies the
 * service's `refresh` (the exact second-stage handler the old
 * `refreshVRAMEstimate` was), keeping the assertions identical.
 */
function makeSchema(seed: number): SchemaNode {
    return {
        type: 'object',
        properties: {
            definition_id: { type: 'string', default: '' },
            max_train_steps: { type: 'integer', default: 1000 + seed },
        },
    } as unknown as SchemaNode;
}

function build() {
    TestBed.configureTestingModule({
        imports: [TrainingDynamicConfigComponent],
        providers: [
            { provide: DatasetService, useValue: { listDatasets: () => of([]) } },
            { provide: DatasetStore, useValue: {} },
            { provide: ToastService, useValue: { error: () => {}, success: () => {} } },
            { provide: SystemService, useValue: {} },
            { provide: JobService, useValue: { estimate: () => of(null) } },
            { provide: ConfigHelpService, useValue: { getConfigHelp: () => of({}) } },
            { provide: ModelService, useValue: { getGlobalSettings: () => of({ default_model_path: '' }) } },
            { provide: RegistryStore, useValue: {} },
            { provide: ModelCapabilitiesService, useValue: { getCapabilities: () => of(null) } },
        ],
    });
    TestBed.overrideComponent(TrainingDynamicConfigComponent, { set: { template: '', imports: [] } });
    const fixture = TestBed.createComponent(TrainingDynamicConfigComponent);
    fixture.componentRef.setInput('schema', makeSchema(0));
    fixture.detectChanges(); // runs the effect → first buildForm()
    return { fixture, component: fixture.componentInstance };
}

describe('TrainingDynamicConfig — form rebuild does not stack valueChanges subscriptions', () => {
    it('only the CURRENT form generation triggers the VRAM-estimate side effect after two rebuilds', () => {
        vi.useFakeTimers();
        try {
            const { fixture, component: c } = build();
            const vramSpy = vi.spyOn(c.vramEstimation, 'refresh');

            const formGen1 = c.form;

            // Rebuild #1 — new schema reference triggers the effect, buildForm()
            // replaces `this.form` with a fresh FormGroup.
            fixture.componentRef.setInput('schema', makeSchema(1));
            fixture.detectChanges();
            const formGen2 = c.form;
            expect(formGen2).not.toBe(formGen1);

            // Rebuild #2.
            fixture.componentRef.setInput('schema', makeSchema(2));
            fixture.detectChanges();
            const formGen3 = c.form;
            expect(formGen3).not.toBe(formGen2);

            // Flush any debounced call the rebuilds themselves may have queued
            // (each buildForm() calls vramEstimation.schedule() directly, e.g.
            // for the "trigger initial VRAM estimate" step) BEFORE clearing the
            // spy, so the per-write assertions below aren't contaminated by a
            // rebuild-queued call landing after mockClear() and being mistaken
            // for proof a later write's OWN pipeline completed.
            vi.advanceTimersByTime(1700);
            vramSpy.mockClear();

            // Stale writes to the two abandoned generations, well spaced apart
            // so each debounce window settles independently instead of
            // coalescing into a single call. 1700ms comfortably clears the
            // full two-stage pipeline (800ms component debounce + 800ms
            // VramEstimationService trigger debounce = 1600ms worst case).
            formGen1.get('max_train_steps')?.setValue(11);
            vi.advanceTimersByTime(1700);
            formGen2.get('max_train_steps')?.setValue(22);
            vi.advanceTimersByTime(1700);

            // A live write to the current generation.
            formGen3.get('max_train_steps')?.setValue(33);
            vi.advanceTimersByTime(1700);

            // With the bug: 3 calls (one leaked subscription per generation).
            // Fixed: only the current generation's subscription is live.
            expect(vramSpy).toHaveBeenCalledTimes(1);
        } finally {
            vi.useRealTimers();
        }
    });

    it('a single edit on the current form after two rebuilds fires the handler exactly once (not 2-3x)', () => {
        vi.useFakeTimers();
        try {
            const { fixture, component: c } = build();
            const vramSpy = vi.spyOn(c.vramEstimation, 'refresh');

            fixture.componentRef.setInput('schema', makeSchema(1));
            fixture.detectChanges();
            fixture.componentRef.setInput('schema', makeSchema(2));
            fixture.detectChanges();

            // Flush any rebuild-queued debounced call before clearing the spy
            // (see the sibling test's comment above).
            vi.advanceTimersByTime(1700);
            vramSpy.mockClear();
            c.form.get('max_train_steps')?.setValue(99);
            // 1700ms comfortably clears the full two-stage pipeline (800ms
            // component debounce + 800ms VramEstimationService trigger debounce).
            vi.advanceTimersByTime(1700);

            expect(vramSpy).toHaveBeenCalledTimes(1);
        } finally {
            vi.useRealTimers();
        }
    });
});
