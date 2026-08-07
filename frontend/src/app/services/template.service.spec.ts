import { TestBed } from '@angular/core/testing';
import { provideHttpClient, withFetch } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { RuntimeConfigService } from './runtime-config.service';
import { TemplateService, type TemplateDomain } from './template.service';

/**
 * `recordUse` is the one place the usage-counter policy lives, so every domain
 * inherits the same behaviour. These assert on the REQUEST that leaves the
 * service — not on "a method was called" — because the URL is the contract the
 * backend's `/templates/{domain}/{id}/use` route answers.
 */
function setup() {
  TestBed.configureTestingModule({
    providers: [
      provideHttpClient(withFetch()), provideHttpClientTesting(),
      { provide: RuntimeConfigService, useValue: { apiUrl: '/api' } },
    ],
  });
  return {
    svc: TestBed.inject(TemplateService),
    http: TestBed.inject(HttpTestingController),
  };
}

const DOMAINS: TemplateDomain[] = ['captioning', 'masking', 'training', 'adaptive'];

describe('TemplateService.recordUse', () => {
  it.each(DOMAINS)('POSTs the %s domain\'s use endpoint', domain => {
    const { svc, http } = setup();
    svc.recordUse(domain, 'tpl-1');

    const req = http.expectOne(`/api/templates/${domain}/tpl-1/use`);
    expect(req.request.method).toBe('POST');
    req.flush({ status: 'recorded' });
    http.verify();
  });

  it('swallows a failure so a lost counter tick never reaches the user', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const { svc, http } = setup();

    // No try/catch here on purpose: an unhandled error would fail this test,
    // which is exactly the regression being pinned.
    svc.recordUse('training', 'tpl-1');
    http.expectOne('/api/templates/training/tpl-1/use')
      .flush({ detail: 'nope' }, { status: 500, statusText: 'Server Error' });

    // Surfaced, not silent (invariant 4) — and named, so a reader of the
    // console knows which domain lost the tick.
    expect(warn).toHaveBeenCalled();
    expect(String(warn.mock.calls[0][0])).toContain('training');
    warn.mockRestore();
  });
});
