import { TestBed } from '@angular/core/testing';
import {
    HttpClient,
    HttpContext,
    HttpErrorResponse,
    provideHttpClient,
    withInterceptors,
} from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import {
    RETRY_ON_TRANSIENT,
    SERVER_UNREACHABLE_MESSAGE,
    transientErrorInterceptor,
} from './transient-error.interceptor';

const FAKE_TIMERS = ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval', 'Date'] as const;

describe('transientErrorInterceptor', () => {
    let http: HttpClient;
    let mock: HttpTestingController;

    beforeEach(() => {
        TestBed.configureTestingModule({
            providers: [
                provideHttpClient(withInterceptors([transientErrorInterceptor])),
                provideHttpClientTesting(),
            ],
        });
        http = TestBed.inject(HttpClient);
        mock = TestBed.inject(HttpTestingController);
    });

    afterEach(() => {
        // RxJS timer() keeps rescheduling against fake clocks — always restore
        // real timers before verifying outstanding requests.
        vi.useRealTimers();
        mock.verify();
    });

    it('retries a GET after a transient status-0 failure, then succeeds', async () => {
        vi.useFakeTimers({ toFake: [...FAKE_TIMERS] });
        let result: unknown;
        let errored: unknown;
        http.get('/api/thing').subscribe({ next: (r) => (result = r), error: (e) => (errored = e) });

        mock.expectOne('/api/thing').error(new ProgressEvent('error')); // status 0
        await vi.advanceTimersByTimeAsync(1000);
        mock.expectOne('/api/thing').flush({ ok: true });

        expect(result).toEqual({ ok: true });
        expect(errored).toBeUndefined();
    });

    it('maps a persistent status-0 failure to a friendly server-unreachable error', async () => {
        vi.useFakeTimers({ toFake: [...FAKE_TIMERS] });
        let errored: HttpErrorResponse | undefined;
        http.get('/api/thing').subscribe({ error: (e) => (errored = e) });

        mock.expectOne('/api/thing').error(new ProgressEvent('error'));
        await vi.advanceTimersByTimeAsync(1000);
        mock.expectOne('/api/thing').error(new ProgressEvent('error'));
        await vi.advanceTimersByTimeAsync(2000);
        mock.expectOne('/api/thing').error(new ProgressEvent('error'));

        expect(errored).toBeInstanceOf(HttpErrorResponse);
        expect(errored!.status).toBe(0);
        expect((errored!.error as { detail: string }).detail).toBe(SERVER_UNREACHABLE_MESSAGE);
    });

    it('does not auto-retry a POST without opt-in, but still maps the error', () => {
        let errored: HttpErrorResponse | undefined;
        http.post('/api/thing', {}).subscribe({ error: (e) => (errored = e) });

        mock.expectOne('/api/thing').error(new ProgressEvent('error'));

        expect(errored!.status).toBe(0);
        expect((errored!.error as { detail: string }).detail).toBe(SERVER_UNREACHABLE_MESSAGE);
        // No second attempt — afterEach's mock.verify() asserts nothing is pending.
    });

    it('retries a POST that opts into RETRY_ON_TRANSIENT', async () => {
        vi.useFakeTimers({ toFake: [...FAKE_TIMERS] });
        let result: unknown;
        const context = new HttpContext().set(RETRY_ON_TRANSIENT, true);
        http.post('/api/import/peek', {}, { context }).subscribe((r) => (result = r));

        mock.expectOne('/api/import/peek').error(new ProgressEvent('error'));
        await vi.advanceTimersByTimeAsync(1000);
        mock.expectOne('/api/import/peek').flush({ kind: 'project' });

        expect(result).toEqual({ kind: 'project' });
    });

    it('passes a non-transient error (400) through unchanged and never retries', () => {
        let errored: HttpErrorResponse | undefined;
        http.get('/api/thing').subscribe({ error: (e) => (errored = e) });

        mock.expectOne('/api/thing').flush(
            { detail: 'bad input' },
            { status: 400, statusText: 'Bad Request' },
        );

        expect(errored!.status).toBe(400);
        expect((errored!.error as { detail: string }).detail).toBe('bad input');
    });
});
