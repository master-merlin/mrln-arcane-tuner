import { TestBed } from '@angular/core/testing';
import { HttpClientTestingModule, HttpTestingController } from '@angular/common/http/testing';
import { ConfigHelpService } from './config-help.service';

// URL assertion carried over byte-identical from the pre-split JobService
// (P4b extraction) — this is the P4b URL-pin guard for the domain split into
// ConfigHelpService (BL2 item 4).
describe('ConfigHelpService — config_help.json static asset (P4b extraction, split from JobService)', () => {
    let svc: ConfigHelpService;
    let http: HttpTestingController;

    beforeEach(() => {
        TestBed.configureTestingModule({
            imports: [HttpClientTestingModule],
            providers: [ConfigHelpService],
        });
        svc = TestBed.inject(ConfigHelpService);
        http = TestBed.inject(HttpTestingController);
    });

    afterEach(() => http.verify());

    it('getConfigHelp() GETs the static /config_help.json asset (not under /api)', () => {
        svc.getConfigHelp().subscribe();
        const req = http.expectOne('/config_help.json');
        expect(req.request.method).toBe('GET');
        req.flush({});
    });
});
