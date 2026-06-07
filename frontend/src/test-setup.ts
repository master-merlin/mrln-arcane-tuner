// zone.js Vitest patch: wraps describe/it/beforeEach in a ProxyZone so that
// fakeAsync/tick/waitForAsync keep working under the Vitest runner. zone.js and
// zone.js/testing are loaded ahead of this file via the build polyfills; this
// side-effect import (added in zone.js 0.16.2) installs the Vitest test-framework
// adapter that zone.js otherwise only ships for jasmine/mocha/jest. Without it,
// fakeAsync specs throw "Expected to be running in 'ProxyZone'".
import 'zone.js/plugins/vitest-patch';
import { afterEach } from 'vitest';

// Restore vi.spyOn() spies after every test. Jasmine (the old Karma runner)
// auto-restored spies between specs; Vitest does not unless asked. Several specs
// spy on globals like document.createElement inside one `it` and rely on the
// original being back for the next test's TestBed.createComponent. Mirroring the
// Jasmine contract here keeps those specs leak-free without per-file boilerplate.
afterEach(() => {
  vi.restoreAllMocks();
});

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
