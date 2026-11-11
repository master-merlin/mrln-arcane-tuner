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
        model_family: { type: 'string', default: 'ernie_image' },
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
        // Declared `array` in the schema but built as a flat FormControl<string[]>
        // (see buildForm's layer_checklist branch) — resetFormToDefaults used to
        // call .clear() on it and THROW, killing the whole reset AND leaking the
        // selector's suppressAutoSave=true forever (auto-create never fired again).
        targeted_layers: {
            type: 'array',
            ui_type: 'layer_checklist',
            default: [],
            items: { type: 'string' },
        },
    },
} as unknown as SchemaNode;

function build(schema: SchemaNode = SCHEMA) {
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
    fixture.componentRef.setInput('schema', schema);
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

    it('keeps the selected model (definition_id AND model_family) on reset', () => {
        // The virtual Default means "default settings for the CURRENT model" —
        // its selector entry carries the current definition id. Resetting
        // model_family to the schema default silently switched the family
        // (ernie_image), which reloads the schema, rebuilds the whole form and
        // detaches every rendered dataset row from the live form.
        const c = build();
        c.form.get('definition_id')?.setValue('flux2-klein-base-9b');
        c.form.get('model_family')?.setValue('flux2');

        c.resetFormToDefaults();

        expect(c.form.get('definition_id')?.value).toBe('flux2-klein-base-9b');
        expect(c.form.get('model_family')?.value).toBe('flux2');
    });

    it('keeps the flux3 selection (flux3-image-klein-base: 50 steps, CFG 3.5, 1024, flow_match) on reset', () => {
        const c = build();
        c.form.get('definition_id')?.setValue('flux3-image-klein-base');
        c.form.get('model_family')?.setValue('flux3');

        c.resetFormToDefaults();

        expect(c.form.get('definition_id')?.value).toBe('flux3-image-klein-base');
        expect(c.form.get('model_family')?.value).toBe('flux3');
    });

    it('keeps the moss_image selection (moss-image-base: 50 steps, CFG 3.0, 1024, flow_match) on reset', () => {
        const c = build();
        c.form.get('definition_id')?.setValue('moss-image-base');
        c.form.get('model_family')?.setValue('moss_image');

        c.resetFormToDefaults();

        expect(c.form.get('definition_id')?.value).toBe('moss-image-base');
        expect(c.form.get('model_family')?.value).toBe('moss_image');
    });

    it('does not throw on layer_checklist arrays (flat FormControl) and resets their value', () => {
        const c = build();
        c.form.get('targeted_layers')?.setValue(['blocks.0', 'blocks.1']);

        expect(() => c.resetFormToDefaults()).not.toThrow();
        expect(c.form.get('targeted_layers')?.value).toEqual([]);
        // The keys AFTER the checklist in schema order must still be reset —
        // previously the throw aborted the loop mid-way.
        expect(c.getFormArray('datasets').length).toBe(1);
    });

    it('places the fixed_sigmas list right after the timestep mode in Training Dynamics', () => {
        const c = build();
        c.organizeGroups([
            { key: 'logit_normal_mu', schema: { group: 'STRATEGY' } },
            { key: 'fixed_sigmas', schema: { group: 'STRATEGY', type: 'array' } },
            { key: 'max_train_steps', schema: { group: 'STRATEGY' } },
            { key: 'timestep_sampling', schema: { group: 'STRATEGY' } },
        ] as unknown as Parameters<typeof c.organizeGroups>[0]);
        const strategy = c.groups().find(g => g.name === 'Training Dynamics');
        expect(strategy?.props.map(p => p.key)).toEqual([
            'max_train_steps', 'timestep_sampling', 'fixed_sigmas', 'logit_normal_mu',
        ]);
    });

    it('releases the selector auto-save suppression even if the reset throws', () => {
        vi.useFakeTimers();
        try {
            const c = build();
            const selector = { suppressAutoSave: { set: vi.fn() } };
            (c as unknown as { templateSelector: unknown }).templateSelector = selector;
            vi.spyOn(c, 'resetFormToDefaults').mockImplementation(() => { throw new Error('boom'); });

            expect(() => c.onTemplateApplied({ config: {}, isDefault: true })).toThrow('boom');
            vi.advanceTimersByTime(1500);

            // suppressAutoSave must NOT stay latched on — that silently disables
            // every future auto-save (the "no template ever created" bug).
            expect(selector.suppressAutoSave.set).toHaveBeenCalledWith(false);
        } finally {
            vi.useRealTimers();
        }
    });
});

/**
 * MiniMax-H3-Image is distilled to a fixed 6-step grid, so its family default draw is
 * `fixed_sigmas` over that grid. The backend derives the descriptor's `defaults` from the
 * definition's `timestep.default_mode` / `timestep.fixed_sigmas` (resolve_capabilities); the
 * form must submit that default when untouched, and the user's own mode when they pick one.
 */
const H3_IMAGE_SIGMAS = [1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132];

// resolve_capabilities(minimax-h3-image-t2i)["defaults"], as the backend returns it.
const H3_IMAGE_DEFAULTS: Record<string, unknown> = {
    resolution: 1024,
    scheduler: 'flow_match',
    learning_rate: 1e-4,
    mixed_precision: 'bf16',
    num_inference_steps: 6,
    guidance_scale: 1.0,
    timestep_sampling: 'fixed_sigmas',
    fixed_sigmas: H3_IMAGE_SIGMAS,
};

const TIMESTEP_SCHEMA: SchemaNode = {
    type: 'object',
    properties: {
        ...(SCHEMA.properties as Record<string, unknown>),
        timestep_sampling: {
            type: 'string',
            enum: ['logit_normal', 'uniform', 'sigmoid', 'cosmap', 'mode', 'flux_shift', 'radc', 'model_shift', 'fixed_sigmas'],
            default: 'logit_normal',
            group: 'STRATEGY',
        },
        fixed_sigmas: {
            anyOf: [{ type: 'array', items: { type: 'number' } }, { type: 'null' }],
            default: null,
            group: 'STRATEGY',
            depends_on: 'timestep_sampling:fixed_sigmas',
        },
    },
} as unknown as SchemaNode;

function submitted(c: ReturnType<typeof build>): Record<string, unknown> {
    let payload: Record<string, unknown> | undefined;
    const sub = c.configSubmitted.subscribe((cfg: Record<string, unknown>) => { payload = cfg; });
    c.onSubmit();
    sub.unsubscribe();
    if (!payload) throw new Error('the form did not submit');
    return payload;
}

function h3Form() {
    const c = build(TIMESTEP_SCHEMA);
    c.form.get('definition_id')?.setValue('minimax-h3-image-t2i');
    c.form.get('model_family')?.setValue('minimax_h3_image');
    c.patchFormRecursive(c.form, { datasets: [{ dataset_name: 'foxes', num_repeats: 1, masking_enabled: false }] });
    c.applyDefinitionDefaults(H3_IMAGE_DEFAULTS);
    return c;
}

describe('TrainingDynamicConfig: the MiniMax-H3-Image timestep default', () => {
    it('an untouched form submits the family default fixed_sigmas', () => {
        const c = h3Form();
        expect(c.form.get('timestep_sampling')?.value).toBe('fixed_sigmas');
        const payload = submitted(c);
        expect(payload['timestep_sampling']).toBe('fixed_sigmas');
        expect(payload['fixed_sigmas']).toEqual(H3_IMAGE_SIGMAS);
    });

    it('an untouched form shows the six sigmas in the list field, not an empty list', () => {
        const c = h3Form();
        expect(c.getFormArray('fixed_sigmas').value).toEqual(H3_IMAGE_SIGMAS);
        expect(c.getFormArray('fixed_sigmas').dirty).toBe(false);
    });

    it('a list the user edited survives the defaults arriving again', () => {
        const c = h3Form();
        const list = c.getFormArray('fixed_sigmas');
        list.removeAt(5);
        list.at(0).setValue(0.9);
        list.markAsDirty();
        c.applyDefinitionDefaults(H3_IMAGE_DEFAULTS);
        expect(list.value).toEqual([0.9, ...H3_IMAGE_SIGMAS.slice(1, 5)]);
    });

    it('a list a template or job restored is kept when the defaults arrive', () => {
        const c = build(TIMESTEP_SCHEMA);
        c.patchFormRecursive(c.form, { timestep_sampling: 'fixed_sigmas', fixed_sigmas: [0.8, 0.4] });
        c.applyDefinitionDefaults(H3_IMAGE_DEFAULTS);
        expect(c.getFormArray('fixed_sigmas').value).toEqual([0.8, 0.4]);
    });

    it('seeding the list does not re-arm template auto-save or mark the form dirty', () => {
        const c = build(TIMESTEP_SCHEMA);
        const selector = { suppressAutoSave: Object.assign(vi.fn(() => false), { set: vi.fn() }) };
        (c as unknown as { templateSelector: unknown }).templateSelector = selector;
        c.applyDefinitionDefaults(H3_IMAGE_DEFAULTS);
        expect(c.getFormArray('fixed_sigmas').value).toEqual(H3_IMAGE_SIGMAS);
        expect(c.form.dirty).toBe(false);
        expect(selector.suppressAutoSave.set).toHaveBeenNthCalledWith(1, true);
        expect(selector.suppressAutoSave.set).toHaveBeenLastCalledWith(false);
    });

    it('a mode the user picks is submitted instead, without the list', () => {
        const c = h3Form();
        const mode = c.form.get('timestep_sampling')!;
        mode.setValue('logit_normal');
        mode.markAsDirty();
        const payload = submitted(c);
        expect(payload['timestep_sampling']).toBe('logit_normal');
        expect('fixed_sigmas' in payload).toBe(false);
    });

    it('a user-picked mode survives the defaults arriving again', () => {
        const c = h3Form();
        const mode = c.form.get('timestep_sampling')!;
        mode.setValue('logit_normal');
        mode.markAsDirty();
        c.applyDefinitionDefaults(H3_IMAGE_DEFAULTS);
        expect(submitted(c)['timestep_sampling']).toBe('logit_normal');
    });
});

describe('TrainingDynamicConfig: array defaults from a definition', () => {
    it('a family with no array default keeps every array as the schema built it', () => {
        const c = build();
        c.applyDefinitionDefaults({ resolution: 1024, learning_rate: 1e-4, scheduler: 'flow_match' });
        expect(c.getFormArray('resolutions').value).toEqual([1024]);
        expect(c.getFormArray('datasets').length).toBe(1);
        expect(c.getFormArray('sample_prompts').length).toBe(0);
        expect(c.form.get('targeted_layers')?.value).toEqual([]);
    });

    it('a primitive list still at its schema default takes the definition default', () => {
        const c = build();
        c.applyDefinitionDefaults({ resolutions: [768, 512] });
        expect(c.getFormArray('resolutions').value).toEqual([768, 512]);
    });

    it('a primitive list that differs from its schema default is never overwritten', () => {
        const c = build();
        c.patchFormRecursive(c.form, { resolutions: [1024, 512] });
        c.applyDefinitionDefaults({ resolutions: [768] });
        expect(c.getFormArray('resolutions').value).toEqual([1024, 512]);
    });

    it('object lists and non-list values for a list are ignored', () => {
        const c = build();
        c.applyDefinitionDefaults({
            datasets: [{ dataset_name: 'x', num_repeats: 2, masking_enabled: true }, { dataset_name: 'y' }],
            resolutions: 768,
        });
        expect(c.getFormArray('datasets').length).toBe(1);
        expect(c.getFormArray('datasets').at(0).value.dataset_name).toBe('');
        expect(c.getFormArray('resolutions').value).toEqual([1024]);
    });
});
