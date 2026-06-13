/**
 * Zoneless-compatible replacement for fakeAsync's `tick()` hygiene drains.
 *
 * One macrotask hop: resolves after the entire pending microtask queue
 * (chained .then/.catch/await continuations) has run. Use it after
 * `fixture.detectChanges()` / `TestBed.tick()` to let fire-and-forget promise
 * chains (e.g. `void sync.refreshDataset(...)`) settle inside the test, before
 * afterEach tears the fixture down.
 *
 * Note: these untracked promises are invisible to `fixture.whenStable()`, so
 * awaiting stability would NOT wait for them — this helper is the right tool.
 */
export const settle = () => new Promise<void>(resolve => setTimeout(resolve));
