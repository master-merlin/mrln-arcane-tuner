import { TestBed } from '@angular/core/testing';
import { FormControl } from '@angular/forms';

import { DynamicFormFieldComponent } from './dynamic-form-field';
import { ToastService } from '../../../services/toast';
import { LoraToolsService } from '../../../services/lora-tools.service';
import { FilesystemService } from '../../../services/filesystem.service';

describe('DynamicFormFieldComponent — sciHint() scientific-notation overlay', () => {
  let component: DynamicFormFieldComponent;

  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [DynamicFormFieldComponent],
      providers: [
        { provide: ToastService, useValue: {} },
        { provide: LoraToolsService, useValue: {} },
        { provide: FilesystemService, useValue: {} },
      ],
    });
    const fixture = TestBed.createComponent(DynamicFormFieldComponent);
    component = fixture.componentInstance;
  });

  it('formats a typical LR into compact scientific notation', () => {
    expect(component.sciHint(0.0001)).toBe('1e-4');
    expect(component.sciHint(0.00015)).toBe('1.5e-4');
    expect(component.sciHint(0.000005)).toBe('5e-6');
  });

  it('accepts string values from the form control', () => {
    expect(component.sciHint('0.0001')).toBe('1e-4');
  });

  it('suppresses the hint when scientific notation adds no clarity (exponent 0)', () => {
    expect(component.sciHint(1.0)).toBe('');
    expect(component.sciHint(5)).toBe('');
  });

  it('shows the hint for large-exponent values too', () => {
    expect(component.sciHint(0.01)).toBe('1e-2');
    expect(component.sciHint(2000)).toBe('2e+3');
  });

  it('hides the hint for empty / zero / non-finite values', () => {
    expect(component.sciHint(0)).toBe('');
    expect(component.sciHint('')).toBe('');
    expect(component.sciHint(null)).toBe('');
    expect(component.sciHint(undefined)).toBe('');
    expect(component.sciHint('abc')).toBe('');
    expect(component.sciHint(NaN)).toBe('');
  });
});

describe('DynamicFormFieldComponent — stepMin() spinner grid alignment', () => {
  function fieldWith(schema: Record<string, unknown>): DynamicFormFieldComponent {
    TestBed.resetTestingModule(); // allow a fresh module per call (multiple per test)
    TestBed.configureTestingModule({
      imports: [DynamicFormFieldComponent],
      providers: [
        { provide: ToastService, useValue: {} },
        { provide: LoraToolsService, useValue: {} },
        { provide: FilesystemService, useValue: {} },
      ],
    });
    const fixture = TestBed.createComponent(DynamicFormFieldComponent);
    fixture.componentRef.setInput('schema', schema);
    return fixture.componentInstance;
  }

  it('anchors the grid to a step-multiple when min is off-grid (max_train_steps)', () => {
    // min 1 / step 100 → arrows must land on 5900/6000, not 5901/6001.
    expect(fieldWith({ type: 'integer', min: 1, step: 100 }).stepMin()).toBe(0);
  });

  it('leaves an already-aligned min untouched', () => {
    expect(fieldWith({ type: 'integer', min: 256, step: 64 }).stepMin()).toBe(256);
    expect(fieldWith({ type: 'integer', min: 0, step: 50 }).stepMin()).toBe(0);
  });

  it('handles fractional steps without floating-point drift', () => {
    // 0.50 / 0.05 = 10 (but 9.999… in IEEE754) — must stay 0.50, not collapse to 0.45.
    expect(fieldWith({ type: 'number', min: 0.5, step: 0.05 }).stepMin()).toBe(0.5);
  });

  it('returns min unchanged when there is no step', () => {
    expect(fieldWith({ type: 'integer', min: 7 }).stepMin()).toBe(7);
  });

  it('returns null when there is no min', () => {
    expect(fieldWith({ type: 'integer', step: 10 }).stepMin()).toBeNull();
  });
});

describe('DynamicFormFieldComponent — string[] field (control_images)', () => {
  function fieldWith(schema: Record<string, unknown>, control: FormControl) {
    TestBed.resetTestingModule();
    TestBed.configureTestingModule({
      imports: [DynamicFormFieldComponent],
      providers: [
        { provide: ToastService, useValue: {} },
        { provide: LoraToolsService, useValue: {} },
        { provide: FilesystemService, useValue: {} },
      ],
    });
    const fixture = TestBed.createComponent(DynamicFormFieldComponent);
    fixture.componentRef.setInput('schema', schema);
    fixture.componentRef.setInput('control', control);
    fixture.componentRef.setInput('fieldKey', 'control_images');
    return fixture;
  }

  const STR_ARRAY = { type: 'array', items: { type: 'string' }, title: 'Control Images' };

  it('detects a plain string[] schema and renders the comma-separated input', () => {
    const fixture = fieldWith(STR_ARRAY, new FormControl([]));
    fixture.detectChanges();
    const c = fixture.componentInstance;
    expect(c.isStringArray()).toBe(true);
    const input = fixture.nativeElement.querySelector('[data-testid="config-input-control_images"]');
    expect(input).toBeTruthy();
    expect(input.getAttribute('type')).toBe('text');
  });

  it('is NOT a string[] for layer_checklist arrays or scalar/number fields', () => {
    expect(fieldWith({ type: 'array', items: { type: 'string' }, ui_type: 'layer_checklist' },
      new FormControl([])).componentInstance.isStringArray()).toBe(false);
    expect(fieldWith({ type: 'integer' }, new FormControl(0)).componentInstance.isStringArray()).toBe(false);
    expect(fieldWith({ type: 'array', items: { type: 'integer' } }, new FormControl([]))
      .componentInstance.isStringArray()).toBe(false);
  });

  it('seeds the text mirror from an existing string[] control value', () => {
    const fixture = fieldWith(STR_ARRAY, new FormControl(['a/before.png', 'b/before.png']));
    fixture.detectChanges(); // ngOnInit
    expect(fixture.componentInstance.stringArrayText()).toBe('a/before.png, b/before.png');
  });

  it('tolerates a legacy "" / null seed (renders empty, no crash)', () => {
    const fixture = fieldWith(STR_ARRAY, new FormControl(''));
    fixture.detectChanges();
    expect(fixture.componentInstance.stringArrayText()).toBe('');
  });

  it('parses comma-separated text back to a trimmed, empty-stripped string[]', () => {
    const control = new FormControl<unknown>([]);
    const c = fieldWith(STR_ARRAY, control).componentInstance;
    c.onStringArrayChange({ target: { value: '  a.png ,, b.png ,  ' } } as unknown as Event);
    expect(control.value).toEqual(['a.png', 'b.png']);
    expect(control.dirty).toBe(true);
  });

  it('clears to an empty array (not "") when the input is emptied', () => {
    const control = new FormControl<unknown>(['x.png']);
    const c = fieldWith(STR_ARRAY, control).componentInstance;
    c.onStringArrayChange({ target: { value: '' } } as unknown as Event);
    expect(control.value).toEqual([]);
  });
});
