// Zoneless: no zone.js anywhere. TestBed defaults to zoneless change detection
// when zone.js is absent, matching the app's provideZonelessChangeDetection().
// fakeAsync/tick/flushMicrotasks are unavailable — specs use async bodies with
// the settle() helper (src/testing/async.ts), TestBed.tick(), or Vitest fake
// timers instead.
import { afterEach } from 'vitest';

// Restore vi.spyOn() spies after every test. Jasmine (the old Karma runner)
// auto-restored spies between specs; Vitest does not unless asked. Several specs
// spy on globals like document.createElement inside one `it` and rely on the
// original being back for the next test's TestBed.createComponent. Mirroring the
// Jasmine contract here keeps those specs leak-free without per-file boilerplate.
afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  failOnLeakedGlobalMock();
});

// LANE-75 guard (class T). `ng test` runs EVERY spec file in one shared
// environment (`--isolate` defaults to false in the Angular builder), so a
// mock assigned straight onto a global (`URL.createObjectURL = vi.fn()`)
// survives vi.restoreAllMocks() — which only undoes vi.spyOn — and becomes the
// NEXT file's problem: its vi.spyOn reuses the leaked mock and inherits the
// call count (CI run 33678676250, LANE-64 run 3). After restore, none of these
// download/clipboard/fetch seams may still be a mock; if one is, the test that
// leaked it fails here, not a victim two files later. Add a seam when a spec
// starts mocking it; never remove one to make a leaking spec pass.
function failOnLeakedGlobalMock(): void {
  const seams: Array<[string, unknown]> = [
    ['URL.createObjectURL', URL.createObjectURL],
    ['URL.revokeObjectURL', URL.revokeObjectURL],
    ['document.createElement', document.createElement],
    ['HTMLAnchorElement.prototype.click', HTMLAnchorElement.prototype.click],
    ['navigator.clipboard.writeText', navigator.clipboard?.writeText],
    ['globalThis.fetch', globalThis.fetch],
  ];
  const leaked = seams.filter(([, value]) => vi.isMockFunction(value)).map(([name]) => name);
  if (leaked.length > 0) {
    throw new Error(
      `Leaked global mock after test: ${leaked.join(', ')}. ` +
        'Use vi.spyOn(...) (restored automatically) or restore the original in your afterEach; ' +
        'spec files share one environment, so a leaked mock breaks the next file.',
    );
  }
}

// Global jsdom shims for browser APIs jsdom does not implement but components reference.
// The Angular unit-test builder initializes the TestBed automatically; do NOT init it here.
// Add further stubs ONLY when a migrated test actually needs them.

if (typeof globalThis.matchMedia !== 'function') {
  globalThis.matchMedia = (query: string) =>
    ({
      matches: false,
      media: query,
      onchange: null,
      addEventListener: () => {},
      removeEventListener: () => {},
      addListener: () => {},
      removeListener: () => {},
      dispatchEvent: () => false,
    }) as unknown as MediaQueryList;
}

if (typeof globalThis.ResizeObserver === 'undefined') {
  globalThis.ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  } as unknown as typeof ResizeObserver;
}
