import { TestBed } from '@angular/core/testing';
import { of } from 'rxjs';

import { TrainingDynamicConfigComponent, type DistillationDescriptor } from './training-dynamic-config';
import { DatasetService } from '../../../services/dataset';
import { DatasetStore } from '../../../state/dataset.store';
import { ToastService } from '../../../services/toast';
import { SystemService } from '../../../services/system.service';
import { JobService } from '../../../services/job';
import { ConfigHelpService } from '../../../services/config-help.service';
import { ModelService } from '../../../services/model.service';
import { RegistryStore } from '../../../state/registry.store';
import { ModelCapabilitiesService, type ModelCapabilities } from '../../../services/model-capabilities.service';
import type { SchemaNode, SchemaProp } from '../schema-node';

/**
 * Distillation handling on the Training form. A definition with a `distillation` block gets
 * `capabilities.distillation` and visible distillation fields from `resolve_capabilities`; a
 * definition without one gets the fields hidden (`field_visibility`) and nothing else. The form:
 *
 *  - shows "Distillation handling" only for a distilled definition, with only the handlings
 *    that definition allows, set to its `default_handling`;
 *  - shows the guidance target and schedule only with contrastive guidance or both, the
 *    target starting at the handling's own default (3.5 alone, 2.5 with the adapter);
 *  - submits the chosen values, and nothing of it for a model that is not distilled;
 *  - submits `constant` as the schedule when nobody touched it, which is what the backend
 *    resolves an unset schedule to.
 */

// The three fields exactly as BaseTrainingConfig.model_json_schema() emits them.
const DISTILLATION_PROPS = {
    distillation_handling: {
        anyOf: [{ enum: ['adapter', 'contrastive_guidance', 'both', 'none'], type: 'string' }, { type: 'null' }],
        default: null,
        group: 'STRATEGY',
        title: 'Distillation handling',
    },
    distillation_guidance_scale: {
        anyOf: [{ type: 'number' }, { type: 'null' }],
        default: null,
        depends_on: 'distillation_handling:contrastive_guidance,both',
        group: 'STRATEGY',
        max: 10.0,
        min: 1.0,
        step: 0.1,
        title: 'Guidance target',
    },
    distillation_guidance_schedule: {
        anyOf: [{ enum: ['constant', 'sigma'], type: 'string' }, { type: 'null' }],
        default: null,
        depends_on: 'distillation_handling:contrastive_guidance,both',
        group: 'STRATEGY',
        title: 'Guidance schedule',
    },
};

const SCHEMA: SchemaNode = {
    type: 'object',
    properties: {
        definition_id: { type: 'string', default: '' },
        max_train_steps: { type: 'integer', default: 1000, group: 'STRATEGY' },
        ...DISTILLATION_PROPS,
    },
} as unknown as SchemaNode;

const FIELDS = ['distillation_handling', 'distillation_guidance_scale', 'distillation_guidance_schedule'];

function descriptor(handlings: string[], defaultHandling: string): DistillationDescriptor {
    const scales: Record<string, number> = {};
    if (handlings.includes('contrastive_guidance')) scales['contrastive_guidance'] = 3.5;
    if (handlings.includes('both')) scales['both'] = 2.5;
    return {
        step: true,
        guidance: handlings.includes('contrastive_guidance'),
        handlings,
        default_handling: defaultHandling,
        has_training_adapter: handlings.includes('adapter'),
        guidance_scale_defaults: scales,
        guidance_schedules: ['constant', 'sigma'],
        default_guidance_schedule: 'constant',
    };
}

function caps(distillation: DistillationDescriptor | null): ModelCapabilities {
    const supported = distillation !== null;
    const defaults: Record<string, unknown> = {};
    if (distillation) {
        defaults['distillation_handling'] = distillation.default_handling;
        defaults['distillation_guidance_schedule'] = 'constant';
        const scale = distillation.guidance_scale_defaults[distillation.default_handling];
        if (scale !== undefined) defaults['distillation_guidance_scale'] = scale;
    }
    return {
        enriched: false,
        block_topology: [],
        lora_targetable_modules: [],
        trainable_layers: [],
        archetype: 'latent_diffusion',
        capabilities: {
            supports_distillation_handling: supported,
            ...(distillation ? { distillation } : {}),
        } as unknown as ModelCapabilities['capabilities'],
        field_visibility: Object.fromEntries(
            FIELDS.map((f) => [f, supported ? { supported: true } : { supported: false, reason: 'this model is not distilled' }]),
        ),
        defaults,
    };
}

function withDefaults(c: ModelCapabilities, extra: Record<string, unknown>): ModelCapabilities {
    return { ...c, defaults: { ...c.defaults, ...extra } };
}

const BY_DEFINITION: Record<string, ModelCapabilities> = {
    // resolve_capabilities(minimax-h3-image-t2i): adapter + guidance, default adapter; one
    // ordinary per-definition default besides (max_train_steps), as every family has some.
    'minimax-h3-image-t2i': withDefaults(caps(descriptor(['adapter', 'contrastive_guidance', 'both', 'none'], 'adapter')), {
        max_train_steps: 2000,
    }),
    // A guidance-distilled model without an adapter.
    'cg-only': caps(descriptor(['contrastive_guidance', 'none'], 'contrastive_guidance')),
    'sdxl_base_1.0': caps(null),
};

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
            { provide: RegistryStore, useValue: { loadFor: () => Promise.resolve(), byId: () => () => null } },
            { provide: ModelCapabilitiesService, useValue: { getCapabilities: (id: string) => of(BY_DEFINITION[id] ?? null) } },
        ],
    });
    TestBed.overrideComponent(TrainingDynamicConfigComponent, { set: { template: '', imports: [] } });
    const fixture = TestBed.createComponent(TrainingDynamicConfigComponent);
    fixture.componentRef.setInput('schema', SCHEMA);
    fixture.detectChanges(); // runs the effect → buildForm()
    const c = fixture.componentInstance;
    const select = (id: string) => {
        c.form.get('definition_id')?.setValue(id);
        fixture.detectChanges(); // the capabilities effect loads this definition's descriptor
    };
    return { c, select };
}

function prop(c: TrainingDynamicConfigComponent, key: string): SchemaProp {
    const found = c.properties().find((p) => p.key === key);
    if (!found) throw new Error(`no property ${key}`);
    return found;
}

const shown = (c: TrainingDynamicConfigComponent, key: string) => !c.shouldHideField(prop(c, key).schema, key);

function submitted(c: TrainingDynamicConfigComponent): Record<string, unknown> {
    let payload: Record<string, unknown> | undefined;
    const sub = c.configSubmitted.subscribe((cfg: Record<string, unknown>) => { payload = cfg; });
    c.onSubmit();
    sub.unsubscribe();
    if (!payload) throw new Error('the form did not submit');
    return payload;
}

function pick(c: TrainingDynamicConfigComponent, key: string, value: unknown) {
    const control = c.form.get(key)!;
    control.setValue(value);
    control.markAsDirty();
}

describe('TrainingDynamicConfig: distillation handling', () => {
    it('a distilled definition shows the field, with only its handlings, set to its default', () => {
        const { c, select } = build();
        select('minimax-h3-image-t2i');
        expect(shown(c, 'distillation_handling')).toBe(true);
        expect(c.fieldSchema(prop(c, 'distillation_handling')).enum).toEqual(['adapter', 'contrastive_guidance', 'both', 'none']);
        expect(c.fieldSchema(prop(c, 'distillation_handling')).options_labels).toEqual({
            adapter: 'Training adapter', contrastive_guidance: 'Contrastive guidance', both: 'Both', none: 'None',
        });
        expect(c.form.get('distillation_handling')?.value).toBe('adapter');
        expect(c.form.get('distillation_handling')?.dirty).toBe(false);
    });

    it('a definition without an adapter offers no adapter', () => {
        const { c, select } = build();
        select('cg-only');
        expect(c.fieldSchema(prop(c, 'distillation_handling')).enum).toEqual(['contrastive_guidance', 'none']);
        expect(c.form.get('distillation_handling')?.value).toBe('contrastive_guidance');
        expect(shown(c, 'distillation_guidance_scale')).toBe(true);
        expect(c.form.get('distillation_guidance_scale')?.value).toBe(3.5);
    });

    it('a definition without the block shows nothing new', () => {
        const { c, select } = build();
        select('sdxl_base_1.0');
        for (const key of FIELDS) expect(shown(c, key)).toBe(false);
        expect(c.distillation()).toBeNull();
        expect(c.fieldSchema(prop(c, 'distillation_handling'))).toBe(prop(c, 'distillation_handling').schema);
    });

    it('the guidance fields show only with contrastive guidance or both', () => {
        const { c, select } = build();
        select('minimax-h3-image-t2i');
        const guidance = () => [shown(c, 'distillation_guidance_scale'), shown(c, 'distillation_guidance_schedule')];
        expect(guidance()).toEqual([false, false]);
        pick(c, 'distillation_handling', 'contrastive_guidance');
        expect(guidance()).toEqual([true, true]);
        pick(c, 'distillation_handling', 'both');
        expect(guidance()).toEqual([true, true]);
        pick(c, 'distillation_handling', 'none');
        expect(guidance()).toEqual([false, false]);
    });

    it('the guidance target follows the handling: 3.5 alone, 2.5 with the adapter, constant schedule', () => {
        const { c, select } = build();
        select('minimax-h3-image-t2i');
        pick(c, 'distillation_handling', 'contrastive_guidance');
        expect(c.form.get('distillation_guidance_scale')?.value).toBe(3.5);
        expect(c.form.get('distillation_guidance_schedule')?.value).toBe('constant');
        pick(c, 'distillation_handling', 'both');
        expect(c.form.get('distillation_guidance_scale')?.value).toBe(2.5);
        pick(c, 'distillation_handling', 'contrastive_guidance');
        expect(c.form.get('distillation_guidance_scale')?.value).toBe(3.5);
    });

    it('a guidance target the user typed is kept when the handling changes', () => {
        const { c, select } = build();
        select('minimax-h3-image-t2i');
        pick(c, 'distillation_handling', 'contrastive_guidance');
        pick(c, 'distillation_guidance_scale', 4.2);
        pick(c, 'distillation_handling', 'both');
        expect(c.form.get('distillation_guidance_scale')?.value).toBe(4.2);
    });

    it('an untouched distilled form submits the default handling and no guidance fields', () => {
        const { c, select } = build();
        select('minimax-h3-image-t2i');
        const payload = submitted(c);
        expect(payload['distillation_handling']).toBe('adapter');
        expect('distillation_guidance_scale' in payload).toBe(false);
        expect('distillation_guidance_schedule' in payload).toBe(false);
    });

    it('the chosen values are submitted; an untouched schedule goes out as constant', () => {
        const { c, select } = build();
        select('minimax-h3-image-t2i');
        pick(c, 'distillation_handling', 'both');
        let payload = submitted(c);
        expect(payload['distillation_handling']).toBe('both');
        expect(payload['distillation_guidance_scale']).toBe(2.5);
        expect(payload['distillation_guidance_schedule']).toBe('constant');

        pick(c, 'distillation_guidance_scale', '3.0'); // a typed number arrives as a string
        pick(c, 'distillation_guidance_schedule', 'sigma');
        payload = submitted(c);
        expect(payload['distillation_guidance_scale']).toBe(3);
        expect(payload['distillation_guidance_schedule']).toBe('sigma');
    });

    it('a model that is not distilled submits no distillation fields, even after a distilled one', () => {
        const { c, select } = build();
        select('minimax-h3-image-t2i');
        pick(c, 'distillation_handling', 'both');
        select('sdxl_base_1.0');
        const payload = submitted(c);
        for (const key of FIELDS) expect(key in payload).toBe(false);
    });

    it('the Default template, applied after the model defaults landed, keeps them', () => {
        // On load the template selector applies "Default" by itself; it used to reset every
        // control to the schema default, wiping what the model's defaults had just filled in.
        const { c, select } = build();
        select('minimax-h3-image-t2i');
        expect(c.form.get('distillation_handling')?.value).toBe('adapter');
        expect(c.form.get('max_train_steps')?.value).toBe(2000);
        c.onTemplateApplied({ config: {}, isDefault: true, auto: true });
        expect(c.form.get('distillation_handling')?.value).toBe('adapter');
        expect(c.form.get('distillation_guidance_schedule')?.value).toBe('constant');
        expect(c.form.get('max_train_steps')?.value).toBe(2000);
        expect(c.form.get('distillation_handling')?.dirty).toBe(false);
        expect(submitted(c)['distillation_handling']).toBe('adapter');
    });

    it('a template of its own still overrides the model defaults', () => {
        const { c, select } = build();
        select('minimax-h3-image-t2i');
        c.onTemplateApplied({
            config: { distillation_handling: 'none', max_train_steps: 300 } as never,
            isDefault: false,
        });
        expect(c.form.get('distillation_handling')?.value).toBe('none');
        expect(c.form.get('max_train_steps')?.value).toBe(300);
        const payload = submitted(c);
        expect(payload['distillation_handling']).toBe('none');
        expect(payload['max_train_steps']).toBe(300);
    });

    it('switching back to the distilled model keeps a valid choice; an invalid one becomes the default', () => {
        const { c, select } = build();
        select('minimax-h3-image-t2i');
        pick(c, 'distillation_handling', 'both');
        select('sdxl_base_1.0');
        select('minimax-h3-image-t2i');
        expect(c.form.get('distillation_handling')?.value).toBe('both');
        select('cg-only'); // 'both' is not allowed there
        expect(c.form.get('distillation_handling')?.value).toBe('contrastive_guidance');
    });
});
