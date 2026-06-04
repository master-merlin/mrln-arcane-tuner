import { TestBed } from '@angular/core/testing';
import { ToastService } from '../toast';

describe('ToastService history', () => {
    let service: ToastService;

    beforeEach(() => {
        TestBed.configureTestingModule({ providers: [ToastService] });
        service = TestBed.inject(ToastService);
    });

    it('records every shown toast in history with type, message, and timestamp', () => {
        service.success('saved');
        service.error('boom');
        service.warning('careful');
        service.info('fyi');

        const h = service.history();
        expect(h.length).toBe(4);
        expect(h[0].message).toBe('fyi');
        expect(h[0].type).toBe('info');
        expect(h[3].message).toBe('saved');
        expect(h[3].type).toBe('success');
        expect(typeof h[0].timestamp).toBe('number');
        expect(h[0].timestamp).toBeGreaterThan(0);
    });

    it('caps history at 20 entries, dropping the oldest', () => {
        for (let i = 0; i < 25; i++) service.info(`msg ${i}`);

        const h = service.history();
        expect(h.length).toBe(20);
        expect(h[0].message).toBe('msg 24');
        expect(h[19].message).toBe('msg 5');
    });

    it('keeps a history entry after its live toast is dismissed', () => {
        service.success('keep me');
        const liveId = service.toasts()[0].id;

        service.dismiss(liveId);

        expect(service.toasts().length).toBe(0);
        expect(service.history().length).toBe(1);
        expect(service.history()[0].message).toBe('keep me');
    });
});
