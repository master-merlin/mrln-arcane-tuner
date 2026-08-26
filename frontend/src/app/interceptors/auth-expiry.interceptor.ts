import { HttpErrorResponse, HttpInterceptorFn } from '@angular/common/http';
import { inject } from '@angular/core';
import { catchError, throwError } from 'rxjs';
import { SessionService } from '../services/session.service';

/**
 * Turn an HTTP 401 into a sign-in, not a silent failure.
 *
 * The backend keeps the session in an `HttpOnly` cookie the browser attaches
 * on its own (`auth.py:49-78`, `main.py:338-361`), so the SPA never handles a
 * token — but it also never noticed the cookie going away. Every request
 * simply started failing, and the UI rendered that as empty lists and stuck
 * spinners: the state a user reads as "the app is broken", not as "sign in
 * again".
 *
 * The error is re-thrown rather than swallowed. Callers still need their own
 * failure path — this adds a global reaction, it does not take over error
 * handling.
 */
export const authExpiryInterceptor: HttpInterceptorFn = (req, next) => {
    const session = inject(SessionService);
    return next(req).pipe(
        catchError((err: unknown) => {
            if (err instanceof HttpErrorResponse && err.status === 401) {
                session.markExpired();
            }
            return throwError(() => err);
        }),
    );
};
