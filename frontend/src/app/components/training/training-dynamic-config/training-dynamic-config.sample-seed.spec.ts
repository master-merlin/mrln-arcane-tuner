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

function setCaps(c: TrainingDynamicConfigComponent, defaults: Record<string, unknown>) {
    (c as unknown as { capabilities: { set(v: unknown): void } }).capabilities.set({ defaults });
}

describe('TrainingDynamicConfig — sample-prompt rows are seeded from the definition', () => {
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
        expect(rows.at(0).get('num_inference_steps')!.value).toBeNull();

        rows.at(1).get('num_inference_steps')!.setValue(20);
        rows.at(1).get('num_inference_steps')!.markAsDirty();
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
        c.applyDefinitionDefaults(MOSS);
        rows.at(1).get('guidance_scale')!.setValue(5);
        rows.at(1).get('guidance_scale')!.markAsDirty();

        c.applyDefinitionDefaults({ num_inference_steps: 28, guidance_scale: 4.0, resolution: 1024 });

        expect(rows.at(0).get('num_inference_steps')!.value).toBe(28);
        expect(rows.at(0).get('guidance_scale')!.value).toBe(4.0);
        expect(rows.at(1).get('guidance_scale')!.value).toBe(5);
    });
});
