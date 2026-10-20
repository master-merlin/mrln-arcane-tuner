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
 * LANE-165 verify 3.01: a sample-prompt row shows (and so sends) the selected definition's
 * sampling defaults. The backend schema carries "unset" as null; the form seeds the row from the
 * definition (MOSS-Image: 50 steps / CFG 3.0) and never overwrites a value the user typed.
 */
const SCHEMA = {
    type: 'object',
    properties: {
        definition_id: { type: 'string', default: '' },
        model_family: { type: 'string', default: 'moss_image' },
        sample_prompts: {
            type: 'array',
            items: {
                type: 'object',
                properties: {
                    prompt: { type: 'string', default: '' },
                    num_inference_steps: { anyOf: [{ type: 'integer' }, { type: 'null' }], default: null },
                    guidance_scale: { anyOf: [{ type: 'number' }, { type: 'null' }], default: null },
                    width: { anyOf: [{ type: 'integer' }, { type: 'null' }], default: null },
                },
            },
        },
    },
} as unknown as SchemaNode;

const MOSS = { num_inference_steps: 50, guidance_scale: 3.0, resolution: 1024 };

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
    fixture.componentRef.setInput('schema', SCHEMA);
    fixture.detectChanges();
    return fixture.componentInstance;
}

function setCaps(c: TrainingDynamicConfigComponent, defaults: Record<string, unknown>, optIn = true) {
    (c as unknown as { capabilities: { set(v: unknown): void } }).capabilities.set({
        defaults,
        capabilities: { seed_prompt_defaults: optIn },
    });
}

describe('TrainingDynamicConfig — sample-prompt rows are seeded from the definition', () => {
    it('a definition that did not opt in keeps the fixed 20 / 3.5 / 1024 row', () => {
        const c = build();
        setCaps(c, MOSS, false);
        c.addArrayItem('sample_prompts', SCHEMA.properties!['sample_prompts'].items);
        const row = c.getFormArray('sample_prompts').at(0);
        expect(row.get('num_inference_steps')!.value).toBe(20);
        expect(row.get('guidance_scale')!.value).toBe(3.5);
        expect(row.get('width')!.value).toBe(1024);
        c.applyDefinitionDefaults(MOSS);
        expect(row.get('num_inference_steps')!.value).toBe(20);
        expect(row.get('guidance_scale')!.value).toBe(3.5);
    });

    it('a new row shows the definition steps / guidance / size', () => {
        const c = build();
        setCaps(c, MOSS);
        c.addArrayItem('sample_prompts', SCHEMA.properties!['sample_prompts'].items);
        const row = c.getFormArray('sample_prompts').at(0);
        expect(row.get('num_inference_steps')!.value).toBe(50);
        expect(row.get('guidance_scale')!.value).toBe(3.0);
        expect(row.get('width')!.value).toBe(1024);
    });

    it('a row added before the capabilities load is seeded when they arrive; a typed value is kept', () => {
        const c = build();
        const items = SCHEMA.properties!['sample_prompts'].items;
        c.addArrayItem('sample_prompts', items);
        c.addArrayItem('sample_prompts', items);
        const rows = c.getFormArray('sample_prompts');
        expect(rows.at(0).get('num_inference_steps')!.value).toBe(20); // no opt-in yet: the fixed seed

        rows.at(1).get('num_inference_steps')!.setValue(20);
        rows.at(1).get('num_inference_steps')!.markAsDirty();
        setCaps(c, MOSS);
        c.applyDefinitionDefaults(MOSS);

        expect(rows.at(0).get('num_inference_steps')!.value).toBe(50);
        expect(rows.at(0).get('guidance_scale')!.value).toBe(3.0);
        // the user's explicit 20 survives, as typed
        expect(rows.at(1).get('num_inference_steps')!.value).toBe(20);
        expect(rows.at(1).get('guidance_scale')!.value).toBe(3.0);
    });

    it('a definition change re-seeds a row still holding the previous seed, not a typed one', () => {
        const c = build();
        const items = SCHEMA.properties!['sample_prompts'].items;
        c.addArrayItem('sample_prompts', items);
        c.addArrayItem('sample_prompts', items);
        const rows = c.getFormArray('sample_prompts');
        setCaps(c, MOSS);
        c.applyDefinitionDefaults(MOSS);
        rows.at(1).get('guidance_scale')!.setValue(5);
        rows.at(1).get('guidance_scale')!.markAsDirty();

        const next = { num_inference_steps: 28, guidance_scale: 4.0, resolution: 1024 };
        setCaps(c, next);
        c.applyDefinitionDefaults(next);

        expect(rows.at(0).get('num_inference_steps')!.value).toBe(28);
        expect(rows.at(0).get('guidance_scale')!.value).toBe(4.0);
        expect(rows.at(1).get('guidance_scale')!.value).toBe(5);
    });

    // LANE-165 verify 5 finding 1.01: a template/job value that happens to EQUAL the current seed
    // is the template's value, not the seed's; a definition change must not overwrite it.
    it('a restored template value equal to the current seed survives a definition change', () => {
        const c = build();
        const items = SCHEMA.properties!['sample_prompts'].items;
        c.addArrayItem('sample_prompts', items);
        setCaps(c, MOSS);
        c.applyDefinitionDefaults(MOSS);
        const rows = c.getFormArray('sample_prompts');
        expect(rows.at(0).get('num_inference_steps')!.value).toBe(50);

        // the template restores two rows: one over the seeded row, one grown by the patch
        const restored = { prompt: 'p', num_inference_steps: 50, guidance_scale: 3.0, width: 1024 };
        c.patchFormRecursive(c.form, { sample_prompts: [restored, { ...restored, prompt: 'q' }] });
        expect(rows.length).toBe(2);

        const next = { num_inference_steps: 28, guidance_scale: 4.0, resolution: 768 };
        setCaps(c, next);
        c.applyDefinitionDefaults(next);

        for (const i of [0, 1]) {
            expect(rows.at(i).get('num_inference_steps')!.value).toBe(50);
            expect(rows.at(i).get('guidance_scale')!.value).toBe(3.0);
            expect(rows.at(i).get('width')!.value).toBe(1024);
        }
    });

    it('a restored unset (null) field is still seeded by the next definition', () => {
        const c = build();
        const items = SCHEMA.properties!['sample_prompts'].items;
        c.addArrayItem('sample_prompts', items);
        setCaps(c, MOSS);
        c.applyDefinitionDefaults(MOSS);
        c.patchFormRecursive(c.form, {
            sample_prompts: [{ prompt: 'p', num_inference_steps: null, guidance_scale: 3.0, width: null }],
        });
        const next = { num_inference_steps: 28, guidance_scale: 4.0, resolution: 768 };
        setCaps(c, next);
        c.applyDefinitionDefaults(next);
        const row = c.getFormArray('sample_prompts').at(0);
        expect(row.get('num_inference_steps')!.value).toBe(28);
        expect(row.get('width')!.value).toBe(768);
        expect(row.get('guidance_scale')!.value).toBe(3.0);
    });
});
