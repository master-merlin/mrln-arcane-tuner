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
