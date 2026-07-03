import { ApplicationConfig, inject, provideZonelessChangeDetection, provideAppInitializer } from '@angular/core';
import { provideRouter } from '@angular/router';
import { provideHttpClient, withFetch, withInterceptors } from '@angular/common/http';
import { routes } from './app.routes';
import { RuntimeConfigService } from './services/runtime-config.service';
import { WebSocketService } from './services/websocket.service';
import { transientErrorInterceptor } from './interceptors/transient-error.interceptor';

export const appConfig: ApplicationConfig = {
  providers: [
    provideZonelessChangeDetection(),
    provideRouter(routes),
    provideHttpClient(withFetch(), withInterceptors([transientErrorInterceptor])),
    // Load runtime config first, THEN open the WebSocket — connect() needs
    // rtc.wsUrl, which is only populated after config.load() resolves.
    provideAppInitializer(() => {
      const config = inject(RuntimeConfigService);
      const ws = inject(WebSocketService);
      return config.load().then(() => ws.connect());
    }),
  ]
};
