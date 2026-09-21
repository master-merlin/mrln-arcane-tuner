/**
 * frame_count_golden_frontend — `estimateFrames` against the SHARED golden.
 *
 * LANE-92 VERIFY 4.01: the UI mirrored ingestion's `int(duration * fps)` and
 * with it the truncation of 4.999999999999998 to 4. The expected counts are
 * DATA, hand-written in `frame-count.golden.json`, and the backend reads the
 * same file (`backend/app/engine/tests/test_frame_count_golden.py`), so the two
 * sides cannot drift apart silently. Nothing here recomputes a count.
 */
import golden from '../../../../testing/golden/frame-count.golden.json';
import { estimateFrames } from './frame-rules';

interface GoldenCase {
    id: string;
    start_s: number;
    end_s: number;
    source_fps: number;
    ingest_fps: number | null;
    frames: number;
    note: string;
}

const CASES = golden.cases as GoldenCase[];

describe('frame_count_golden_frontend', () => {
    it('holds the cases the review named', () => {
        const ids = CASES.map(c => c.id);
        expect(ids).toEqual(expect.arrayContaining([
            'five_frames_from_1s_at_24', '22_frames_from_2s_at_24', 'fraction_floors_0.98s_at_24',
        ]));
        expect(new Set(ids).size).toBe(CASES.length);
        expect(CASES.length).toBeGreaterThanOrEqual(10);
    });

    for (const c of CASES) {
        it(`${c.id}: ${c.frames} frames (${c.note})`, () => {
            expect(estimateFrames(c.start_s, c.end_s, c.source_fps, c.ingest_fps)).toBe(c.frames);
        });
    }
});
