import { TestBed } from '@angular/core/testing';
import { of } from 'rxjs';

import { TrainingDynamicConfigComponent } from './training-dynamic-config';
import { DatasetService } from '../../../services/dataset';
import { DatasetStore } from '../../../state/dataset.store';
import { ToastService } from '../../../services/toast';
import { SystemService } from '../../../services/system.service';
import { JobService } from '../../../services/job';
import { ModelService } from '../../../services/model.service';
import { RegistryStore } from '../../../state/registry.store';
import { ModelCapabilitiesService } from '../../../services/model-capabilities.service';
import { HttpClient } from '@angular/common/http';
import type { SchemaNode } from '../schema-node';

/**
 * Default-template completeness: applying the (virtual) Default template runs
 * `resetFormToDefaults()`. The backend schema declares `datasets` with
 * `minItems: 1` and NO default, so a plain "clear + re-add schema defaults"
 * left the form with ZERO dataset rows — the Datasets card collapsed to its
 * empty state ("No items added") the moment change detection caught up, e.g.
 * when the user clicked the per-dataset "masking enabled" toggle. The reset
 * must keep the form complete: every object-array with a minimum row count
 * gets seeded back to that minimum, exactly like `buildForm()` does on load.
 */
const SCHEMA: SchemaNode = {
    type: 'object',
    properties: {
        definition_id: { type: 'string', default: '' },
        max_train_steps: { type: 'integer', default: 1000 },
        resolutions: { type: 'array', default: [1024], items: { type: 'integer' } },
        datasets: {
            type: 'array',
            minItems: 1,
            items: {
                type: 'object',
                properties: {
                    dataset_name: { type: 'string', default: '' },
                    num_repeats: { type: 'integer', default: 1 },
                    masking_enabled: { type: 'boolean', default: false },
                },
            },
        },
        sample_prompts: {
            type: 'array',
            items: {
                type: 'object',
                properties: { prompt: { type: 'string', default: '' } },
            },
        },
    },
} as unknown as SchemaNode;

function build() {
    TestBed.configureTestingModule({
        imports: [TrainingDynamicConfigComponent],
        providers: [
            { provide: HttpClient, useValue: { get: () => of({}) } },
            { provide: DatasetService, useValue: { listDatasets: () => of([]) } },
            { provide: DatasetStore, useValue: {} },
            { provide: ToastService, useValue: { error: () => {}, success: () => {} } },
            { provide: SystemService, useValue: {} },
            { provide: JobService, useValue: { estimate: () => of(null) } },
            { provide: ModelService, useValue: { getGlobalSettings: () => of({ default_model_path: '' }) } },
            { provide: RegistryStore, useValue: {} },
            { provide: ModelCapabilitiesService, useValue: { getCapabilities: () => of(null) } },
        ],
    });
    TestBed.overrideComponent(TrainingDynamicConfigComponent, { set: { template: '', imports: [] } });
    const fixture = TestBed.createComponent(TrainingDynamicConfigComponent);
    fixture.componentRef.setInput('schema', SCHEMA);
    fixture.detectChanges(); // runs the effect → buildForm()
    return fixture.componentInstance;
}

describe('TrainingDynamicConfig — Default template reset keeps the form complete', () => {
    it('buildForm seeds one dataset row (precondition)', () => {
        const c = build();
        expect(c.getFormArray('datasets').length).toBe(1);
    });

    it('resetFormToDefaults keeps one (blank) dataset row instead of clearing to zero', () => {
        const c = build();
        // Simulate a configured dataset with masking enabled…
        c.patchFormRecursive(c.form, {
            datasets: [{ dataset_name: 'my-set', num_repeats: 3, masking_enabled: true }],
        });
        // …then the user picks the Default template.
        c.resetFormToDefaults();

        const datasets = c.getFormArray('datasets');
        expect(datasets.length).toBe(1);
        expect(datasets.at(0).value).toEqual({
            dataset_name: '', num_repeats: 1, masking_enabled: false,
        });
    });

    it('resetFormToDefaults shrinks a multi-dataset config back to the single seeded row', () => {
        const c = build();
        c.patchFormRecursive(c.form, {
            datasets: [
                { dataset_name: 'a', num_repeats: 1, masking_enabled: false },
                { dataset_name: 'b', num_repeats: 2, masking_enabled: true },
            ],
        });
        expect(c.getFormArray('datasets').length).toBe(2);

        c.resetFormToDefaults();
        expect(c.getFormArray('datasets').length).toBe(1);
    });

    it('arrays without a minimum still reset to empty (sample_prompts)', () => {
        const c = build();
        c.patchFormRecursive(c.form, { sample_prompts: [{ prompt: 'x' }] });
        c.resetFormToDefaults();
        expect(c.getFormArray('sample_prompts').length).toBe(0);
    });

    it('primitive arrays keep their schema defaults on reset (resolutions)', () => {
        const c = build();
        c.patchFormRecursive(c.form, { resolutions: [1024, 768, 512] });
        c.resetFormToDefaults();
        expect(c.getFormArray('resolutions').value).toEqual([1024]);
    });
});
