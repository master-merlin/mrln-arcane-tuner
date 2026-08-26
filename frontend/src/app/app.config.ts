import { ApplicationConfig, inject, provideZonelessChangeDetection, provideAppInitializer } from '@angular/core';
import { provideRouter } from '@angular/router';
import { provideHttpClient, withFetch, withInterceptors } from '@angular/common/http';
import { routes } from './app.routes';
import { RuntimeConfigService } from './services/runtime-config.service';
import { WebSocketService } from './services/websocket.service';
import { transientErrorInterceptor } from './interceptors/transient-error.interceptor';
import { authExpiryInterceptor } from './interceptors/auth-expiry.interceptor';

export const appConfig: ApplicationConfig = {
  providers: [
    provideZonelessChangeDetection(),
    provideRouter(routes),
    // authExpiry is OUTERMOST on purpose: interceptors see the response in
    // reverse order, so it observes the FINAL error after the transient
    // interceptor has exhausted its retries — not each intermediate attempt.
    provideHttpClient(withFetch(), withInterceptors([authExpiryInterceptor, transientErrorInterceptor])),
    // Load runtime config first, THEN open the WebSocket — connect() needs
    // rtc.wsUrl, which is only populated after config.load() resolves.
    provideAppInitializer(() => {
      const config = inject(RuntimeConfigService);
      const ws = inject(WebSocketService);
      return config.load().then(() => ws.connect());
    }),
  ]
};
