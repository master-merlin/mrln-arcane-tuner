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
 * A definition switch re-seeds what the previous definition seeded.
 *
 * The form fills a control from a definition's `defaults` only while it still holds the schema
 * default. Without provenance, MOSS's seeded mu 1.09 then counts as "someone's value" on a switch
 * to FLUX.2 (whose defaults name no mu) and stays; FLUX.3's seeded `uniform` stays on a switch to
 * MOSS. The form remembers what it seeded: a control it still owns takes the new definition's
 * default, or the schema default when the new definition names none. A user's edit and a
 * template's or job's restored value are never owned, so a switch never replaces them, even when
 * they equal a seeded or a schema value, until the Default template's reset hands them back.
 */
const SCHEMA: SchemaNode = {
    type: 'object',
    properties: {
        definition_id: { type: 'string', default: '' },
        model_family: { type: 'string', default: 'flux2' },
        learning_rate: { type: 'number', default: 1e-4 },
        datasets: {
            type: 'array',
            minItems: 1,
            items: { type: 'object', properties: { dataset_name: { type: 'string', default: '' } } },
        },
        timestep_sampling: {
            type: 'string',
            enum: ['logit_normal', 'uniform', 'sigmoid', 'cosmap', 'mode', 'flux_shift', 'radc', 'model_shift', 'fixed_sigmas'],
            default: 'logit_normal',
            group: 'STRATEGY',
        },
        logit_normal_mu: { type: 'number', default: 0.0, group: 'STRATEGY', depends_on: 'timestep_sampling:logit_normal' },
        logit_normal_sigma: { type: 'number', default: 1.0, group: 'STRATEGY', depends_on: 'timestep_sampling:logit_normal' },
        flux_shift_max: { type: 'number', default: 1.16, group: 'STRATEGY', depends_on: 'timestep_sampling:flux_shift' },
        fixed_sigmas: {
            anyOf: [{ type: 'array', items: { type: 'number' } }, { type: 'null' }],
            default: null,
            group: 'STRATEGY',
            depends_on: 'timestep_sampling:fixed_sigmas',
        },
    },
} as unknown as SchemaNode;

// resolve_capabilities(...)["defaults"], as the backend returns them (timestep keys and one other).
const MOSS = { learning_rate: 1e-4, timestep_sampling: 'logit_normal', logit_normal_mu: 1.09, logit_normal_sigma: 0.95 };
const FLUX2 = { learning_rate: 1e-4 };
const FLUX3 = { learning_rate: 1e-4, timestep_sampling: 'uniform' };
const OMNIGEN2 = { learning_rate: 1e-4, timestep_sampling: 'flux_shift', flux_shift_max: 1.15 };
const H3_IMAGE = { learning_rate: 1e-4, timestep_sampling: 'fixed_sigmas', fixed_sigmas: [1.0, 0.75, 0.25] };

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
    fixture.detectChanges(); // runs the effect -> buildForm()
    return fixture.componentInstance;
}

type Form = ReturnType<typeof build>;

const value = (c: Form, key: string) => c.form.get(key)?.value;
const timestep = (c: Form) => ({
    mode: value(c, 'timestep_sampling'), mu: value(c, 'logit_normal_mu'), sigma: value(c, 'logit_normal_sigma'),
});

function submitted(c: Form): Record<string, unknown> {
    let payload: Record<string, unknown> | undefined;
    const sub = c.configSubmitted.subscribe((cfg: Record<string, unknown>) => { payload = cfg; });
    c.onSubmit();
    sub.unsubscribe();
    if (!payload) throw new Error('the form did not submit');
    return payload;
}

/** Apply a template the way the selector does, then let the 1.5 s template guard lapse. */
function applyTemplate(c: Form, config: Record<string, unknown>) {
    c.onTemplateApplied({ config, isDefault: false });
    vi.advanceTimersByTime(1600);
}

describe('TrainingDynamicConfig: a definition switch re-seeds what it seeded', () => {
    beforeEach(() => vi.useFakeTimers());
    afterEach(() => vi.useRealTimers());

    it('MOSS -> FLUX.2 puts mu and sigma back to the schema 0.0 / 1.0', () => {
        const c = build();
        c.applyDefinitionDefaults(MOSS);
        expect(timestep(c)).toEqual({ mode: 'logit_normal', mu: 1.09, sigma: 0.95 });
        c.applyDefinitionDefaults(FLUX2);
        expect(timestep(c)).toEqual({ mode: 'logit_normal', mu: 0.0, sigma: 1.0 });
    });

    it('FLUX.3 -> MOSS gives logit_normal with 1.09 / 0.95', () => {
        const c = build();
        c.applyDefinitionDefaults(FLUX3);
        expect(value(c, 'timestep_sampling')).toBe('uniform');
        c.applyDefinitionDefaults(MOSS);
        expect(timestep(c)).toEqual({ mode: 'logit_normal', mu: 1.09, sigma: 0.95 });
    });

    it('OmniGen2 -> FLUX.2 puts the mode and the shift back to the schema', () => {
        const c = build();
        c.applyDefinitionDefaults(OMNIGEN2);
        c.applyDefinitionDefaults(FLUX2);
        expect(value(c, 'timestep_sampling')).toBe('logit_normal');
        expect(value(c, 'flux_shift_max')).toBe(1.16);
    });

    it('MiniMax-H3-Image -> MOSS drops the seeded sigma list', () => {
        const c = build();
        c.applyDefinitionDefaults(H3_IMAGE);
        expect(c.getFormArray('fixed_sigmas').value).toEqual([1.0, 0.75, 0.25]);
        c.applyDefinitionDefaults(MOSS);
        expect(c.getFormArray('fixed_sigmas').value).toEqual([]);
        expect(timestep(c)).toEqual({ mode: 'logit_normal', mu: 1.09, sigma: 0.95 });
    });

    it('FLUX.3 -> MOSS submits the re-seeded mu and sigma (their mode-gated controls are enabled again)', () => {
        const c = build();
        c.patchFormRecursive(c.form, { datasets: [{ dataset_name: 'foxes' }] });
        c.applyDefinitionDefaults(FLUX3);
        // The switch itself is a form change: it re-evaluates the mode-gated controls (mu and
        // sigma go disabled under uniform) before the new definition's defaults arrive.
        c.form.get('definition_id')!.setValue('moss-image-base');
        expect('logit_normal_mu' in submitted(c)).toBe(false); // gated off under uniform
        c.applyDefinitionDefaults(MOSS);
        const payload = submitted(c);
        expect(payload['timestep_sampling']).toBe('logit_normal');
        expect(payload['logit_normal_mu']).toBe(1.09);
        expect(payload['logit_normal_sigma']).toBe(0.95);
    });

    it('a seeded value marks nothing dirty', () => {
        const c = build();
        c.applyDefinitionDefaults(MOSS);
        c.applyDefinitionDefaults(FLUX2);
        expect(c.form.dirty).toBe(false);
    });
});

describe('TrainingDynamicConfig: a switch never replaces a value that is not its own', () => {
    beforeEach(() => vi.useFakeTimers());
    afterEach(() => vi.useRealTimers());

    it('a hand-set mu 0.5 survives MOSS -> MOSS SFT and MOSS -> FLUX.2', () => {
        const c = build();
        c.applyDefinitionDefaults(MOSS);
        const mu = c.form.get('logit_normal_mu')!;
        mu.setValue(0.5);
        mu.markAsDirty();
        c.applyDefinitionDefaults(MOSS);
        expect(value(c, 'logit_normal_mu')).toBe(0.5);
        c.applyDefinitionDefaults(FLUX2);
        expect(value(c, 'logit_normal_mu')).toBe(0.5);
        expect(value(c, 'logit_normal_sigma')).toBe(1.0); // still seeded, so it follows the switch
    });

    it('a hand-set mu equal to the seed is the user\'s once typed', () => {
        const c = build();
        c.applyDefinitionDefaults(MOSS);
        const mu = c.form.get('logit_normal_mu')!;
        mu.setValue(1.09);
        mu.markAsDirty();
        c.applyDefinitionDefaults(FLUX2);
        expect(value(c, 'logit_normal_mu')).toBe(1.09);
    });

    it('a value another part of the form wrote is no longer the seed\'s', () => {
        const c = build();
        c.applyDefinitionDefaults(MOSS);
        c.form.get('logit_normal_mu')!.setValue(0.7, { emitEvent: false }); // not dirty, not the seed
        c.applyDefinitionDefaults(FLUX2);
        expect(value(c, 'logit_normal_mu')).toBe(0.7);
    });

    it('a template restoring MOSS 1.09 / 0.95 keeps them after 1.5 s and a switch to FLUX.2', () => {
        const c = build();
        applyTemplate(c, { timestep_sampling: 'logit_normal', logit_normal_mu: 1.09, logit_normal_sigma: 0.95 });
        c.applyDefinitionDefaults(MOSS);
        c.applyDefinitionDefaults(FLUX2);
        expect(timestep(c)).toEqual({ mode: 'logit_normal', mu: 1.09, sigma: 0.95 });
    });

    it('a template restoring the schema 0.0 / 1.0 keeps them after 1.5 s and a switch to MOSS', () => {
        const c = build();
        applyTemplate(c, { timestep_sampling: 'logit_normal', logit_normal_mu: 0.0, logit_normal_sigma: 1.0 });
        c.applyDefinitionDefaults(MOSS);
        expect(timestep(c)).toEqual({ mode: 'logit_normal', mu: 0.0, sigma: 1.0 });
    });

    it('a job restore is kept like a template', () => {
        const c = build();
        c.loadExternalConfig({ timestep_sampling: 'uniform' });
        vi.advanceTimersByTime(1600);
        c.applyDefinitionDefaults(MOSS);
        expect(value(c, 'timestep_sampling')).toBe('uniform');
    });

    it('the Default template hands every control back to the seeding', () => {
        const c = build();
        applyTemplate(c, { timestep_sampling: 'logit_normal', logit_normal_mu: 0.0, logit_normal_sigma: 1.0 });
        c.onTemplateApplied({ config: {}, isDefault: true });
        vi.advanceTimersByTime(1600);
        c.applyDefinitionDefaults(MOSS);
        expect(timestep(c)).toEqual({ mode: 'logit_normal', mu: 1.09, sigma: 0.95 });
        c.applyDefinitionDefaults(FLUX2);
        expect(timestep(c)).toEqual({ mode: 'logit_normal', mu: 0.0, sigma: 1.0 });
    });
});
