import { TestBed } from '@angular/core/testing';
import { HttpClientTestingModule, HttpTestingController } from '@angular/common/http/testing';
import { signal } from '@angular/core';
import { of } from 'rxjs';
import { SystemService } from './system.service';
import { RuntimeConfigService } from './runtime-config.service';
import { WebSocketService } from './websocket.service';

describe('SystemService — version + logs (P4b extraction)', () => {
    let svc: SystemService;
    let http: HttpTestingController;

    beforeEach(() => {
        TestBed.configureTestingModule({
            imports: [HttpClientTestingModule],
            providers: [
                SystemService,
                { provide: RuntimeConfigService, useValue: { apiUrl: 'http://test/api' } },
                {
                    provide: WebSocketService,
                    useValue: {
                        isConnected: signal(false),
                        reconnected$: of(),
                        send: () => {},
                        on: () => of(),
                    },
                },
            ],
        });
        svc = TestBed.inject(SystemService);
        http = TestBed.inject(HttpTestingController);
    });

    afterEach(() => http.verify());

    it('getVersion() GETs /system/version', () => {
        svc.getVersion().subscribe();
        const req = http.expectOne('http://test/api/system/version');
        expect(req.request.method).toBe('GET');
        req.flush({ version: '1.2.3' });
    });

    it('getLogs() GETs /system/logs?lines=200 by default', () => {
        svc.getLogs().subscribe();
        const req = http.expectOne('http://test/api/system/logs?lines=200');
        expect(req.request.method).toBe('GET');
        req.flush([]);
    });

    it('getLogs(n) honors an explicit line count', () => {
        svc.getLogs(50).subscribe();
        const req = http.expectOne('http://test/api/system/logs?lines=50');
        req.flush([]);
    });

    it('clearLogs() POSTs /system/logs/clear', () => {
        svc.clearLogs().subscribe();
        const req = http.expectOne('http://test/api/system/logs/clear');
        expect(req.request.method).toBe('POST');
        expect(req.request.body).toEqual({});
        req.flush({ message: 'ok' });
    });
});
