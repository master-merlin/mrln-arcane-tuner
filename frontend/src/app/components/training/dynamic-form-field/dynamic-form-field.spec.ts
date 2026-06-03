import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';

import { DynamicFormFieldComponent } from './dynamic-form-field';
import { ToastService } from '../../../services/toast';
import { RuntimeConfigService } from '../../../services/runtime-config.service';

describe('DynamicFormFieldComponent — sciHint() scientific-notation overlay', () => {
  let component: DynamicFormFieldComponent;

  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [DynamicFormFieldComponent],
      providers: [
        provideHttpClient(),
        { provide: ToastService, useValue: {} },
        { provide: RuntimeConfigService, useValue: { apiUrl: '', mediaBaseUrl: '' } },
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
