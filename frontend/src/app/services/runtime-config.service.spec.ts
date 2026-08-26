import { TestBed } from '@angular/core/testing';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { RuntimeConfigService } from './runtime-config.service';

/**
 * Guards for runtime-config validation (plan Task 3, F-09).
 *
 * The old `load()` did `{ ...DEFAULT_CONFIG, ...data }` on whatever the fetch
 * returned. A non-object body would have spread into indexed characters; an
 * out-of-range port would have been adopted verbatim. Nothing crashed, which is
 * why it survived — the service just held nonsense.
 *
 * `load()` is called from APP_INITIALIZER, so the binding requirement is that it
 * NEVER rejects: a rejected initializer means the app does not bootstrap at all.
 * Every case below asserts that as well as the parsed result.
 */
describe('RuntimeConfigService', () => {
    let service: RuntimeConfigService;
    let warn: ReturnType<typeof vi.spyOn>;

    const mockFetch = (body: unknown, ok = true) => {
        vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
            ok,
            status: ok ? 200 : 500,
            json: () => Promise.resolve(body),
        }));
    };

    beforeEach(() => {
        TestBed.configureTestingModule({ providers: [RuntimeConfigService] });
        service = TestBed.inject(RuntimeConfigService);
        warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    });

    afterEach(() => {
        vi.unstubAllGlobals();
        vi.restoreAllMocks();
        TestBed.resetTestingModule();
    });

    describe('URLs are origin-relative and independent of the config', () => {
        it('derives apiUrl and mediaBaseUrl without consulting the file', () => {
            // This is why the port fields are dead: the backend serves the SPA,
            // so a port in a config file could only ever disagree with the port
            // the page was actually loaded from.
            expect(service.apiUrl).toBe('/api');
            expect(service.mediaBaseUrl).toBe('/media');
        });

        it('derives the ws scheme from the page protocol', () => {
            expect(service.wsUrl).toMatch(/^wss?:\/\/.+\/api\/ws$/);
        });
    });

    describe('malformed bodies are ignored, never adopted', () => {
        for (const [label, body] of [
            ['a string', '"nope"'],
            ['a number', 42],
            ['null', null],
            ['an array', [1, 2, 3]],
            ['a boolean', true],
        ] as const) {
            it(`ignores ${label} without rejecting`, async () => {
                mockFetch(body);
                await expect(service.load()).resolves.toBeUndefined();
                // Still serviceable — the app must bootstrap regardless.
                expect(service.apiUrl).toBe('/api');
            });
        }
    });

    describe('port validation', () => {
        it('accepts a valid deprecated port', async () => {
            mockFetch({ backendPort: 8000, frontendPort: 4200 });
            await expect(service.load()).resolves.toBeUndefined();
            expect(warn).not.toHaveBeenCalled();
        });

        for (const bad of [0, -1, 65536, 99999, 1.5, '8000', null]) {
            it(`rejects out-of-range or non-numeric port ${JSON.stringify(bad)}`, async () => {
                mockFetch({ backendPort: bad });
                await expect(service.load()).resolves.toBeUndefined();
                expect(warn).toHaveBeenCalled();
            });
        }
    });

    describe('unknown keys and transport failures', () => {
        it('ignores unknown keys instead of merging them onto state', async () => {
            mockFetch({ backendPort: 8000, somethingNew: 'ignored', __proto__: {} });
            await expect(service.load()).resolves.toBeUndefined();
            expect((service as unknown as { config: Record<string, unknown> }).config)
                .toEqual({ backendPort: 8000 });
        });

        it('survives a non-OK response', async () => {
            mockFetch({}, false);
            await expect(service.load()).resolves.toBeUndefined();
            expect(service.apiUrl).toBe('/api');
        });

        it('survives a network rejection — APP_INITIALIZER must not reject', async () => {
            vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('offline')));
            await expect(service.load()).resolves.toBeUndefined();
            expect(service.apiUrl).toBe('/api');
        });

        it('survives a body that is not valid JSON', async () => {
            vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
                ok: true,
                status: 200,
                json: () => Promise.reject(new SyntaxError('bad json')),
            }));
            await expect(service.load()).resolves.toBeUndefined();
            expect(service.apiUrl).toBe('/api');
        });
    });
});
