import { TestBed } from '@angular/core/testing';
import { Subject } from 'rxjs';
import { SystemUpdateService, UpdateStatus } from '../system-update.service';
import { RuntimeConfigService } from '../runtime-config.service';
import { WebSocketService } from '../websocket.service';
import { ToastService } from '../toast';
import { HttpClient } from '@angular/common/http';

function status(partial: Partial<UpdateStatus>): UpdateStatus {
    return {
        state: 'idle', available: true, branch: 'main', commit: 'abc1234',
        dirty: false, is_repo: true, behind: 0, active: 0, error: null, ...partial,
    };
}

describe('SystemUpdateService', () => {
    let service: SystemUpdateService;
    let httpGet: ReturnType<typeof vi.fn>;
    let httpPost: ReturnType<typeof vi.fn>;
    let wsEvents: Subject<UpdateStatus>;
    let toastInfo: ReturnType<typeof vi.fn>;

    beforeEach(() => {
        httpGet = vi.fn().mockReturnValue(new Subject());
        httpPost = vi.fn().mockReturnValue(new Subject());
        wsEvents = new Subject<UpdateStatus>();
        toastInfo = vi.fn();

        TestBed.configureTestingModule({
            providers: [
                SystemUpdateService,
                { provide: HttpClient, useValue: { get: httpGet, post: httpPost } },
                { provide: RuntimeConfigService, useValue: { apiUrl: '/api' } },
                { provide: WebSocketService, useValue: { on: () => wsEvents.asObservable() } },
                { provide: ToastService, useValue: { info: toastInfo } },
            ],
        });
        service = TestBed.inject(SystemUpdateService);
    });

    it('updates the status signal from update.status WS events', () => {
        wsEvents.next(status({ state: 'pulling', behind: 2 }));
        expect(service.status()?.state).toBe('pulling');
        expect(service.status()?.behind).toBe(2);
    });

    it('refreshStatus() calls GET /system/update/status', () => {
        service.refreshStatus();
        expect(httpGet).toHaveBeenCalledWith('/api/system/update/status');
    });

    it('check() POSTs to /system/update/check', () => {
        service.check().subscribe();
        expect(httpPost).toHaveBeenCalledWith('/api/system/update/check', {});
    });

    it('apply() POSTs to /system/update/apply', () => {
        service.apply().subscribe();
        expect(httpPost).toHaveBeenCalledWith('/api/system/update/apply', {});
    });

    it('updateReady() is true when available and behind > 0', () => {
        wsEvents.next(status({ behind: 3 }));
        expect(service.updateReady()).toBe(true);
    });

    it('fires a one-time toast on behind 0 -> >0 transition', () => {
        wsEvents.next(status({ behind: 0 }));
        wsEvents.next(status({ behind: 2 }));
        wsEvents.next(status({ behind: 2 }));
        expect(toastInfo).toHaveBeenCalledTimes(1);
    });
});
