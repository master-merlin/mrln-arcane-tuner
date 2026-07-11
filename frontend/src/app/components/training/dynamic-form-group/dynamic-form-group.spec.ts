import { Component, ChangeDetectionStrategy, signal } from '@angular/core';
import { TestBed } from '@angular/core/testing';
import { FormArray, FormControl, FormGroup } from '@angular/forms';
import { of } from 'rxjs';

import { DynamicFormGroupComponent } from './dynamic-form-group';
import { DatasetService } from '../../../services/dataset';
import { DatasetStore } from '../../../state/dataset.store';
import { RuntimeConfigService } from '../../../services/runtime-config.service';
import type { SchemaNode } from '../schema-node';
import type {
    ModelCapabilities,
    FieldVisibility,
} from '../../../services/model-capabilities.service';

/**
 * OnPush staleness: the rows render from `formArray().controls`, but template
 * applies (the on-load auto-apply, Jobs handoffs) mutate the FormArray from
 * async HTTP callbacks — no event inside this view, so nothing marked it for
 * check. The stale rows stayed on screen, bound to detached controls, until
 * the user's next click ran change detection and the rows "vanished" under
 * the cursor (the Training-screen "dataset disappears when I click masking
 * enabled" bug). The component must mark itself for check whenever its
 * FormArray changes, no matter who changed it.
 */
const SCHEMA: SchemaNode = {
    type: 'array',
    items: { type: 'string' },
} as SchemaNode;

@Component({
    standalone: true,
    changeDetection: ChangeDetectionStrategy.OnPush,
    imports: [DynamicFormGroupComponent],
    // Mirrors the real parent (training-dynamic-config): the form reference is
    // re-read whenever the parent is checked, and buildForm() REPLACES it.
    template: `<app-dynamic-form-group fieldKey="my_list" [schema]="schema" [parentForm]="form()" />`,
})
class HostComponent {
    schema = SCHEMA;
    form = signal(new FormGroup({
        my_list: new FormArray([new FormControl('alpha'), new FormControl('beta')]),
    }));
}

function build() {
    TestBed.configureTestingModule({
        imports: [HostComponent],
        providers: [
            { provide: DatasetService, useValue: { listDatasets: () => of([]) } },
            { provide: DatasetStore, useValue: { entities: () => [] } },
            { provide: RuntimeConfigService, useValue: { mediaBaseUrl: '' } },
        ],
    });
    const fixture = TestBed.createComponent(HostComponent);
    fixture.detectChanges();
    return fixture;
}

describe('DynamicFormGroup — re-renders on external FormArray mutations (OnPush)', () => {
    it('renders the initial rows (precondition)', () => {
        const fixture = build();
        const inputs = fixture.nativeElement.querySelectorAll('input');
        expect(inputs.length).toBe(2);
    });

    it('shows the empty state after the array is cleared outside any UI event', () => {
        const fixture = build();
        const arr = fixture.componentInstance.form().get('my_list') as FormArray;

        // Simulate an async template apply clearing the array (no click, no
        // input event — exactly what resetFormToDefaults does from HTTP).
        arr.clear();
        fixture.detectChanges();

        expect(fixture.nativeElement.querySelectorAll('input').length).toBe(0);
        expect(fixture.nativeElement.querySelector('[data-testid="config-array-empty"]')).toBeTruthy();
    });

    it('renders rows grown externally (template apply adding datasets)', () => {
        const fixture = build();
        const arr = fixture.componentInstance.form().get('my_list') as FormArray;

        arr.push(new FormControl('gamma'));
        fixture.detectChanges();

        expect(fixture.nativeElement.querySelectorAll('input').length).toBe(3);
    });

    it('rebinds row inputs when the parent form is REPLACED (schema rebuild)', () => {
        // A definition/family change rebuilds the whole form (buildForm()).
        // Rows tracked by $index reuse their DOM, leaving formControlName
        // directives attached to the OLD form's controls — edits then write
        // into the void and auto-save never sees them (live bug: toggling
        // "masking enabled" saved masking_enabled:false forever).
        const fixture = build();
        const newForm = new FormGroup({
            my_list: new FormArray([new FormControl('fresh-a'), new FormControl('fresh-b')]),
        });
        fixture.componentInstance.form.set(newForm);
        fixture.detectChanges();

        const input: HTMLInputElement = fixture.nativeElement.querySelector('input');
        expect(input.value).toBe('fresh-a');

        // An edit through the DOM must land in the NEW form, not the old one.
        input.value = 'edited';
        input.dispatchEvent(new Event('input'));
        fixture.detectChanges();

        expect((newForm.get('my_list') as FormArray).at(0).value).toBe('edited');
    });

    it('maps inline-group keys to friendly labels', () => {
        const fixture = build();
        const child = fixture.debugElement.children[0].componentInstance as DynamicFormGroupComponent;
        expect(child.inlineGroupLabel('masking_toggles')).toBe('Enable masking');
        expect(child.inlineGroupLabel('caption_toggles')).toBe('Captions');
        expect(child.inlineGroupLabel('future_group')).toBe('Future Group');
    });
});

/**
 * Dataset-row placement: the 3-column grid must read
 * [Captions | Enable masking] [Original Weight] [Mask Opacity]
 * — ALL inline toggle groups share ONE grid cell (side by side), and
 * original_weight is reordered before mask_opacity so they land in
 * columns 2 and 3 of the same row instead of wrapping.
 */
const DATASETS_SCHEMA: SchemaNode = {
    type: 'array',
    items: {
        type: 'object',
        properties: {
            use_captions: { type: 'boolean', default: true, inline_group: 'caption_toggles' },
            use_model_aware_captions: { type: 'boolean', default: true, inline_group: 'caption_toggles', depends_on: 'use_captions' },
            masking_enabled: { type: 'boolean', default: false, inline_group: 'masking_toggles' },
            recreate_masks: { type: 'boolean', default: false, inline_group: 'masking_toggles', depends_on: 'masking_enabled' },
            // Real schema declares mask_opacity BEFORE original_weight; the row
            // wants original_weight (col 2) before mask_opacity (col 3).
            mask_opacity: { type: 'number', default: 0, depends_on: 'masking_enabled' },
            original_weight: { type: 'number', default: 1, depends_on: 'masking_enabled' },
        },
    },
} as SchemaNode;

@Component({
    standalone: true,
    changeDetection: ChangeDetectionStrategy.OnPush,
    imports: [DynamicFormGroupComponent],
    template: `<app-dynamic-form-group fieldKey="datasets" [schema]="schema" [parentForm]="form()" />`,
})
class DatasetsHostComponent {
    schema = DATASETS_SCHEMA;
    form = signal(new FormGroup({
        datasets: new FormArray([
            new FormGroup({
                use_captions: new FormControl(true),
                use_model_aware_captions: new FormControl(true),
                masking_enabled: new FormControl(false),
                recreate_masks: new FormControl(false),
            }),
        ]),
    }));
}

describe('DynamicFormGroup — dataset row toggle placement', () => {
    function buildDatasets() {
        TestBed.configureTestingModule({
            imports: [DatasetsHostComponent],
            providers: [
                { provide: DatasetService, useValue: { listDatasets: () => of([]) } },
                { provide: DatasetStore, useValue: { entities: () => [] } },
                { provide: RuntimeConfigService, useValue: { mediaBaseUrl: '' } },
            ],
        });
        const fixture = TestBed.createComponent(DatasetsHostComponent);
        fixture.detectChanges();
        return fixture;
    }

    it('renders ALL inline toggle groups inside ONE shared grid cell', () => {
        const fixture = buildDatasets();
        const cells = fixture.nativeElement.querySelectorAll('[data-testid="config-inline-groups-cell"]');
        expect(cells.length).toBe(1);
        // Both group headers live in the same cell, side by side.
        expect(cells[0].textContent).toContain('Captions');
        expect(cells[0].textContent).toContain('Enable masking');
        // All four toggles are present.
        expect(cells[0].querySelectorAll('input[type="checkbox"]').length).toBe(4);
    });

    it('orders original_weight before mask_opacity in the datasets row', () => {
        const fixture = buildDatasets();
        const child = fixture.debugElement.children[0].componentInstance as DynamicFormGroupComponent;
        const keys = child.nestedProps().map(p => p.key);
        expect(keys.indexOf('original_weight')).toBeLessThan(keys.indexOf('mask_opacity'));
    });
});

/**
 * Nested-field capability gating (W4-3). Per-dataset fields render from a
 * capability-BLIND branch — until this fix, `field_visibility` was consulted
 * only for TOP-LEVEL fields, so a per-dataset field the family doesn't support
 * (e.g. `masking_enabled` on a paired edit model, gated by
 * archetypes.py `supports_masking_variants`) still rendered a live toggle that
 * did nothing. The group must consult the SAME `field_visibility` descriptor
 * for nested props: gated-off ⇒ not rendered (same UX as top-level), and a
 * hidden master toggle takes its `depends_on` dependents with it so no orphaned
 * greyed control lingers.
 */
function nestedCaps(
    overrides: Record<string, FieldVisibility>,
): ModelCapabilities {
    return {
        enriched: true,
        block_topology: [],
        lora_targetable_modules: [],
        trainable_layers: [],
        archetype: 'edit',
        capabilities: {} as ModelCapabilities['capabilities'],
        field_visibility: overrides,
        defaults: {},
    };
}

@Component({
    standalone: true,
    changeDetection: ChangeDetectionStrategy.OnPush,
    imports: [DynamicFormGroupComponent],
    template: `<app-dynamic-form-group fieldKey="datasets" [schema]="schema" [parentForm]="form()" [capabilities]="caps()" />`,
})
class GatedDatasetsHostComponent {
    schema = DATASETS_SCHEMA;
    caps = signal<ModelCapabilities | null>(null);
    form = signal(new FormGroup({
        datasets: new FormArray([
            new FormGroup({
                use_captions: new FormControl(true),
                use_model_aware_captions: new FormControl(true),
                masking_enabled: new FormControl(false),
                recreate_masks: new FormControl(false),
                mask_opacity: new FormControl(0),
                original_weight: new FormControl(1),
            }),
        ]),
    }));
}

describe('DynamicFormGroup — nested capability gating', () => {
    function buildGated(caps: ModelCapabilities | null) {
        TestBed.configureTestingModule({
            imports: [GatedDatasetsHostComponent],
            providers: [
                { provide: DatasetService, useValue: { listDatasets: () => of([]) } },
                { provide: DatasetStore, useValue: { entities: () => [] } },
                { provide: RuntimeConfigService, useValue: { mediaBaseUrl: '' } },
            ],
        });
        const fixture = TestBed.createComponent(GatedDatasetsHostComponent);
        fixture.componentInstance.caps.set(caps);
        fixture.detectChanges();
        return fixture;
    }

    const maskId = '[data-testid="config-nested-checkbox-datasets-masking_enabled"]';
    const recreateId = '[data-testid="config-nested-checkbox-datasets-recreate_masks"]';
    const captionsId = '[data-testid="config-nested-checkbox-datasets-use_captions"]';

    it('renders the masking toggle for a supporting family (no descriptor)', () => {
        const el = buildGated(null).nativeElement as HTMLElement;
        expect(el.querySelector(maskId)).toBeTruthy();
        expect(el.querySelector(captionsId)).toBeTruthy();
    });

    it('renders the masking toggle when the descriptor marks it supported', () => {
        const el = buildGated(
            nestedCaps({ masking_enabled: { supported: true } }),
        ).nativeElement as HTMLElement;
        expect(el.querySelector(maskId)).toBeTruthy();
    });

    it('HIDES the masking toggle for a family that gates it off (edit model)', () => {
        const el = buildGated(
            nestedCaps({ masking_enabled: { supported: false, reason: 'paired edit' } }),
        ).nativeElement as HTMLElement;
        // The gated master toggle is gone…
        expect(el.querySelector(maskId)).toBeNull();
        // …and its depends_on dependent (recreate_masks) goes with it — no
        // orphaned, permanently-disabled control under an "Enable masking" header.
        expect(el.querySelector(recreateId)).toBeNull();
        // Unrelated groups (Captions) are untouched.
        expect(el.querySelector(captionsId)).toBeTruthy();
    });

    it('drops the whole "Enable masking" inline group when it empties out', () => {
        const el = buildGated(
            nestedCaps({ masking_enabled: { supported: false } }),
        ).nativeElement as HTMLElement;
        const cell = el.querySelector('[data-testid="config-inline-groups-cell"]');
        expect(cell).toBeTruthy(); // caption_toggles still renders the cell
        expect(cell!.textContent).toContain('Captions');
        expect(cell!.textContent).not.toContain('Enable masking');
    });

    it('leaves nested rendering unchanged when no field is gated off', () => {
        const el = buildGated(
            nestedCaps({ some_unrelated_field: { supported: false } }),
        ).nativeElement as HTMLElement;
        const cell = el.querySelector('[data-testid="config-inline-groups-cell"]');
        expect(cell!.querySelectorAll('input[type="checkbox"]').length).toBe(4);
    });
});

/**
 * `resolutions` renders the specialized preset picker (presets + custom +
 * add) — not the generic "Add Item" primitive-array list — and keeps its
 * original (unscoped) test ids.
 */
const RES_INT_SCHEMA: SchemaNode = {
    type: 'array',
    items: { type: 'integer' },
} as SchemaNode;

@Component({
    standalone: true,
    changeDetection: ChangeDetectionStrategy.OnPush,
    imports: [DynamicFormGroupComponent],
    template: `<app-dynamic-form-group [fieldKey]="key" [schema]="schema" [parentForm]="form()" />`,
})
class ResPickerHostComponent {
    key = 'resolutions';
    schema = RES_INT_SCHEMA;
    form = signal(new FormGroup({
        resolutions: new FormArray<FormControl<number | null>>([new FormControl(1024)]),
    }));
}

describe('DynamicFormGroup — resolutions uses the resolution picker', () => {
    function buildRes(key: string) {
        TestBed.configureTestingModule({
            imports: [ResPickerHostComponent],
            providers: [
                { provide: DatasetService, useValue: { listDatasets: () => of([]) } },
                { provide: DatasetStore, useValue: { entities: () => [] } },
                { provide: RuntimeConfigService, useValue: { mediaBaseUrl: '' } },
            ],
        });
        const fixture = TestBed.createComponent(ResPickerHostComponent);
        fixture.componentInstance.key = key;
        fixture.detectChanges();
        return fixture;
    }

    it('leaves the resolutions picker ids unscoped and at the default 2-col span', () => {
        const el = buildRes('resolutions').nativeElement as HTMLElement;
        // Original, unscoped ids preserved (no existing selector breaks).
        expect(el.querySelector('[data-testid="config-res-preset-1024"]')).toBeTruthy();
        expect(el.querySelector('[data-testid="config-add-custom-res-btn"]')).toBeTruthy();
        const root = el.querySelector('div') as HTMLElement;
        expect(root.classList.contains('md:col-span-2')).toBe(true);
        expect(root.classList.contains('col-span-full')).toBe(false);
    });
});
