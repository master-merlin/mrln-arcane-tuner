import { HttpContextToken, HttpErrorResponse, HttpInterceptorFn } from '@angular/common/http';
import { catchError, retry, throwError, timer } from 'rxjs';

/**
 * Statuses that mean "the server isn't reachable right now" rather than a real
 * application error: 0 (no HTTP response at all — connection refused/reset, e.g.
 * the RunPod proxy hitting a backend that's mid-restart or still booting) plus
 * the gateway 5xx that a reverse proxy emits while the upstream is down.
 */
const TRANSIENT_STATUSES = new Set([0, 502, 503, 504]);

/** HTTP methods safe to replay automatically (idempotent reads). */
const SAFE_METHODS = new Set(['GET', 'HEAD', 'OPTIONS']);

/**
 * Opt-in flag for requests that are idempotent reads despite not using a safe
 * method — e.g. the import wizard's `POST /import/peek` and `/import/plan`,
 * which only inspect the uploaded archive. Set it on the request's HttpContext
 * to make the interceptor auto-retry it on a transient failure. The mutating
 * `/import/apply` step deliberately does NOT set it.
 */
export const RETRY_ON_TRANSIENT = new HttpContextToken<boolean>(() => false);

/** Number of retries (in addition to the first attempt) with 1s, 2s backoff. */
const MAX_RETRIES = 2;

/** User-facing message surfaced when the backend can't be reached. */
export const SERVER_UNREACHABLE_MESSAGE =
    "Can't reach the server — it may be restarting. Wait a few seconds and try again.";

function isTransient(err: unknown): err is HttpErrorResponse {
    return err instanceof HttpErrorResponse && TRANSIENT_STATUSES.has(err.status);
}

/**
 * Recover from transient backend-unreachable failures: auto-retry idempotent
 * requests with a short backoff, and rewrite any surviving transient error into
 * a friendly, envelope-shaped message ({@link SERVER_UNREACHABLE_MESSAGE} under
 * `error.detail`) so callers show useful text instead of Angular's raw
 * "Http failure response for …: 0 undefined". Non-transient errors (4xx, real
 * 500s) pass through untouched.
 */
export const transientErrorInterceptor: HttpInterceptorFn = (req, next) => {
    const retryable = SAFE_METHODS.has(req.method) || req.context.get(RETRY_ON_TRANSIENT);
    return next(req).pipe(
        retry({
            count: retryable ? MAX_RETRIES : 0,
            // Only delay-and-retry transient failures; anything else aborts the
            // retry loop immediately by re-throwing.
            delay: (err, attempt) =>
                isTransient(err) ? timer(attempt * 1000) : throwError(() => err),
        }),
        catchError((err: unknown) =>
            isTransient(err)
                ? throwError(
                      () =>
                          new HttpErrorResponse({
                              error: {
                                  detail: SERVER_UNREACHABLE_MESSAGE,
                                  error_code: 'SERVER_UNREACHABLE',
                              },
                              status: err.status,
                              statusText: err.statusText || 'Server Unreachable',
                              url: err.url ?? undefined,
                          }),
                  )
                : throwError(() => err),
        ),
    );
};
