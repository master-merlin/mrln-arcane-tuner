import { describe, it, expect } from 'vitest';
import {
    adaptEvents,
    latestAdaptState,
    newestAdaptState,
    lossSeries,
    logTail,
    type AdaptEvent,
} from './job-metrics';

const stepLine = (step: number, extra: object) =>
    JSON.stringify({ step, status: 'training', loss: 0.5, progress: 1, ...extra });
const adaptLine = (data: object) => JSON.stringify({ adapt: data });

describe('adaptive metrics parsing', () => {
    it('adaptEvents extracts adapt payloads in order and ignores step logs', () => {
        const logs = [
            stepLine(10, { adaptive_active: 8 }),
            adaptLine({ step: 20, kind: 'narrow', active_count: 5, total_count: 8 }),
            adaptLine({ step: 40, kind: 'probe_open', active_count: 8, total_count: 8 }),
            'plain human log line',
        ];
        const events = adaptEvents(logs);
        expect(events.map((e) => e.kind)).toEqual(['narrow', 'probe_open']);
    });

    it('latestAdaptState returns the newest event or null', () => {
        expect(latestAdaptState([])).toBeNull();
        const logs = [adaptLine({ step: 20, kind: 'narrow', active_count: 5, total_count: 8 })];
        expect(latestAdaptState(logs)?.active_count).toBe(5);
    });

    it('newestAdaptState: higher step wins, null-tolerant, ties to b (LANE-35)', () => {
        const a: AdaptEvent = { step: 1450, kind: 'narrow', active_count: 181, total_count: 224 };
        const b: AdaptEvent = { step: 1700, kind: 'narrow', active_count: 176, total_count: 224 };
        expect(newestAdaptState(null, null)).toBeNull();
        expect(newestAdaptState(null, b)).toBe(b);
        expect(newestAdaptState(a, null)).toBe(a);
        expect(newestAdaptState(a, b)).toBe(b); // durable older, live newer
        expect(newestAdaptState(b, a)).toBe(b); // live older (evicted refetch)
        expect(newestAdaptState(a, { ...a })).not.toBe(a); // tie → b
        expect(newestAdaptState(a, { ...a })?.step).toBe(1450);
    });

    it('lossSeries carries adaptive fields through for charting', () => {
        const pts = lossSeries([stepLine(10, { adaptive_active: 6, adaptive_hot: 3 })], 0);
        expect(pts[0].adaptive_active).toBe(6);
        expect(pts[0].adaptive_hot).toBe(3);
    });

    it('adapt lines are never mistaken for step metrics', () => {
        // adapt payloads have step but no loss/status → must not enter lossSeries
        const pts = lossSeries([adaptLine({ step: 20, kind: 'narrow', active_count: 5, total_count: 8 })], 0);
        expect(pts).toEqual([]);
    });

    it('adapt lines never leak into the plain log tail as raw JSON', () => {
        // Prove the negative: a human-log-bearing stream with an adapt event
        // mixed in must surface only the human lines — the JSON line must not
        // become a visible "log line" in the tail.
        const logs = [
            'plain human log line one',
            adaptLine({ step: 20, kind: 'narrow', active_count: 5, total_count: 8 }),
            'plain human log line two',
        ];
        const tail = logTail(logs);
        expect(tail.map((l) => l.text)).toEqual(['plain human log line one', 'plain human log line two']);
    });
});
