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
