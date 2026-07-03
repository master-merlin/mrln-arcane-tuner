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
import type { SchemaNode } from '../schema-node';

/**
 * Template-load fidelity: applying a saved template MUST restore every field
 * back into the reactive form — scalars, nested object-arrays (`datasets`) and
 * primitive arrays (`resolutions`). The reported failure was a primitive array
 * that loaded only its first element ("only 1024 selected") because the
 * FormArray was never grown before `patchValue` (which silently ignores values
 * beyond the array's current length).
 */
const SCHEMA: SchemaNode = {
    type: 'object',
    properties: {
        definition_id: { type: 'string', default: '' },
        model_family: { type: 'string', default: '' },
        max_train_steps: { type: 'integer', default: 1000 },
        learning_rate: { type: 'number', default: 0.0001 },
        resolution_strategy: { type: 'string', default: 'single' },
        resolutions: { type: 'array', default: [1024], items: { type: 'integer' } },
        datasets: {
            type: 'array',
            items: {
                type: 'object',
                properties: {
                    dataset_name: { type: 'string', default: '' },
                    num_repeats: { type: 'integer', default: 1 },
                },
            },
        },
    },
} as unknown as SchemaNode;

const CONFIG = {
    definition_id: 'flux2-klein-base-9b',
    model_family: 'flux2',
    max_train_steps: 5000,
    learning_rate: 0.0002,
    resolution_strategy: 'mixed',
    resolutions: [1024, 768, 512, 1440],
    datasets: [{ dataset_name: 'Porsche 935-78 Moby Dick - 1978', num_repeats: 3 }],
};

function build() {
    TestBed.configureTestingModule({
        imports: [TrainingDynamicConfigComponent],
        providers: [
            { provide: DatasetService, useValue: { listDatasets: () => of([]) } },
            { provide: DatasetStore, useValue: {} },
            { provide: ToastService, useValue: { error: () => {}, success: () => {} } },
            { provide: SystemService, useValue: {} },
            { provide: JobService, useValue: { estimate: () => of(null), getConfigHelp: () => of({}) } },
            { provide: ModelService, useValue: { getGlobalSettings: () => of({ default_model_path: '' }) } },
            { provide: RegistryStore, useValue: {} },
            { provide: ModelCapabilitiesService, useValue: { getCapabilities: () => of(null) } },
        ],
    });
    // Empty template: we exercise the form-patching logic, not the heavy
    // child-component tree (which would need its own inputs/services).
    TestBed.overrideComponent(TrainingDynamicConfigComponent, { set: { template: '', imports: [] } });
    const fixture = TestBed.createComponent(TrainingDynamicConfigComponent);
    fixture.componentRef.setInput('schema', SCHEMA);
    fixture.detectChanges(); // runs the effect → buildForm()
    return fixture.componentInstance;
}

describe('TrainingDynamicConfig — template load fidelity (patchFormRecursive)', () => {
    it('builds the form from schema defaults (resolutions seeded with [1024])', () => {
        const c = build();
        expect(c.getFormArray('resolutions').value).toEqual([1024]);
        expect(c.form.get('max_train_steps')?.value).toBe(1000);
    });

    it('restores ALL fields from an applied template config', () => {
        const c = build();
        c.patchFormRecursive(c.form, CONFIG);
        const out = c.form.getRawValue();
        expect(out.definition_id).toBe('flux2-klein-base-9b');
        expect(out.max_train_steps).toBe(5000);
        expect(out.learning_rate).toBe(0.0002);
        expect(out.resolution_strategy).toBe('mixed');
        // The crux: the full resolutions list, not just the seeded [1024].
        expect(out.resolutions).toEqual([1024, 768, 512, 1440]);
        expect(out.datasets).toEqual([
            { dataset_name: 'Porsche 935-78 Moby Dick - 1978', num_repeats: 3 },
        ]);
    });

    it('grows a primitive array even when the schema is not resolvable mid-load', () => {
        const c = build();
        // Simulate the schema/properties not being populated yet (the mid-load
        // race that previously dropped fields). resolutions is seeded [1024].
        c.properties.set([]);
        c.patchFormRecursive(c.form, { resolutions: [1024, 768, 512, 1440] });
        expect(c.getFormArray('resolutions').value).toEqual([1024, 768, 512, 1440]);
    });

    it('grows an EMPTY primitive array mid-load (worst case: no seed, no schema)', () => {
        const c = build();
        // Drain the seeded entry so the array starts empty, and blank the schema.
        c.getFormArray('resolutions').clear();
        c.properties.set([]);
        c.patchFormRecursive(c.form, { resolutions: [768, 512] });
        expect(c.getFormArray('resolutions').value).toEqual([768, 512]);
    });
});
