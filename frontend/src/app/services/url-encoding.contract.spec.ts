import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { DatasetService } from './dataset';
import { JobService } from './job';
import { RuntimeConfigService } from './runtime-config.service';
import { datasetPreviewUrl } from '../shared/media-preview';

/**
 * Guard for the URL path-segment encoding contract (plan Task 2, RULE-20
 * class T). Contract: `_harness/research/url-path-segment-encoding-contract.md`,
 * ECOSYSTEM §5 row 4.
 *
 * The defect these pin is not theoretical. Raw interpolation of a filename
 * containing `#` or `?` truncates the URL at the fragment or query, so the
 * backend receives a DIFFERENT filename and answers 200 for it — the caption
 * editor then reads and writes the wrong sidecar with no error anywhere. A
 * dataset containing `shot#3.png` was silently broken.
 *
 * These assert on the URL the HTTP layer actually receives, not on a helper's
 * return value, because the call site is where the bug lived.
 */

const API = 'http://localhost:8000/api';

/** Characters that break a raw interpolation, and ones that must survive it. */
const HOSTILE = [
    { label: 'fragment marker', value: 'shot#3.png', mustContain: '%23' },
    { label: 'query marker', value: 'what?.png', mustContain: '%3F' },
    { label: 'percent', value: '100%.png', mustContain: '%25' },
    { label: 'space', value: 'a shot.png', mustContain: '%20' },
    { label: 'non-ASCII', value: 'äöü-日本.png', mustContain: '%C3%A4' },
    { label: 'ampersand', value: 'a&b.png', mustContain: '%26' },
    { label: 'plus', value: 'a+b.png', mustContain: '%2B' },
];

describe('URL path-segment encoding contract', () => {
    let http: HttpTestingController;

    beforeEach(() => {
        TestBed.configureTestingModule({
            providers: [
                provideHttpClient(),
                provideHttpClientTesting(),
                { provide: RuntimeConfigService, useValue: { apiUrl: API, wsUrl: 'ws://x/ws' } },
            ],
        });
        http = TestBed.inject(HttpTestingController);
    });

    afterEach(() => {
        http.verify();
        TestBed.resetTestingModule();
    });

    describe('dataset caption sidecars', () => {
        for (const c of HOSTILE) {
            it(`encodes ${c.label} in getCaption`, () => {
                const svc = TestBed.inject(DatasetService);
                svc.getCaption('ds', c.value).subscribe();
                const req = http.expectOne(r => r.url.includes('/captions/'));
                expect(req.request.url).toContain(c.mustContain);
                // The decisive assertion: nothing after the encoded segment,
                // i.e. the URL was not truncated at `#` or `?`.
                expect(req.request.url).toBe(
                    `${API}/datasets/ds/captions/${encodeURIComponent(c.value)}`,
                );
                req.flush({ content: '' });
            });
        }

        it('encodes the filename in saveCaption too — the write path, not just the read', () => {
            const svc = TestBed.inject(DatasetService);
            svc.saveCaption('ds', 'shot#3.png', 'hello').subscribe();
            const req = http.expectOne(r => r.method === 'PUT');
            expect(req.request.url).toBe(`${API}/datasets/ds/captions/shot%233.png`);
            req.flush({ status: 'ok' });
        });

        it('encodes lyrics sidecars on the same contract', () => {
            const svc = TestBed.inject(DatasetService);
            svc.getLyrics('ds', 'track#1.txt').subscribe();
            const req = http.expectOne(r => r.url.includes('/lyrics/'));
            expect(req.request.url).toBe(`${API}/datasets/ds/lyrics/track%231.txt`);
            req.flush({ content: '' });
        });
    });

    describe('nested relative paths still survive encoding', () => {
        it('keeps a sub-directory path addressable through the :path converter', () => {
            // Encoding a `/` to `%2F` is transparent: the ASGI server decodes
            // before Starlette matches, so the converter sees the slashes.
            const svc = TestBed.inject(DatasetService);
            svc.getCaption('ds', 'sub/nested.png').subscribe();
            const req = http.expectOne(r => r.url.includes('/captions/'));
            expect(req.request.url).toBe(`${API}/datasets/ds/captions/sub%2Fnested.png`);
            req.flush({ content: '' });
        });
    });

    describe('job ids', () => {
        it('encodes a hostile job id', () => {
            const svc = TestBed.inject(JobService);
            svc.getJobLogs('job#1').subscribe();
            const req = http.expectOne(r => r.url.includes('/logs'));
            expect(req.request.url).toBe(`${API}/jobs/job%231/logs`);
            req.flush([]);
        });

        it('does NOT encode a pre-built query string', () => {
            // `${q}` at job.ts:286 is already-encoded query text. Encoding it
            // would escape the `?` and turn the query into a path segment.
            const svc = TestBed.inject(JobService);
            svc.restartJob('job-1', true).subscribe();
            const req = http.expectOne(r => r.method === 'POST');
            // The `?` must stay a literal query delimiter, NOT become %3F.
            expect(req.request.url).toBe(`${API}/jobs/job-1/restart?fresh=true`);
            expect(req.request.url).not.toContain('%3F');
            req.flush({ status: 'ok', fresh: true });
        });
    });

    describe('media preview URLs', () => {
        it('encodes the relative media path in BOTH branches', () => {
            // Regression: the query branch encoded this value while the direct
            // branch did not — one function, one variable, two treatments. A
            // `#` left raw truncates the URL at the fragment.
            const thumb = datasetPreviewUrl(API, `${API}/media`, 'ds', 'sub/shot#3.png');
            expect(thumb).toContain('%23');
            expect(thumb).not.toMatch(/#3\.png$/);

            // GIFs take the direct branch — the one that used to leak the `#`.
            const direct = datasetPreviewUrl(API, `${API}/media`, 'ds', 'sub/loop#3.gif');
            expect(direct).toContain('%23');
            expect(direct).not.toMatch(/#3\.gif$/);
        });
    });
});
