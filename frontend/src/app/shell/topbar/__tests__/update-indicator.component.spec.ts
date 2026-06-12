import { TestBed } from '@angular/core/testing';
import { signal } from '@angular/core';
import { UpdateIndicatorComponent } from '../update-indicator.component';
import { SystemUpdateService } from '../../../services/system-update.service';
import { TopbarPanelStore } from '../../../state/topbar-panel.store';

describe('UpdateIndicatorComponent', () => {
    function setup(opts: { available: boolean; ready: boolean; behind: number }) {
        const svc = {
            available: signal(opts.available),
            updateReady: signal(opts.ready),
            isBusy: signal(false),
            status: signal({ behind: opts.behind, commit: 'abc1234', branch: 'main' }),
            apply: vi.fn().mockReturnValue({ subscribe: vi.fn() }),
            check: vi.fn().mockReturnValue({ subscribe: vi.fn() }),
        };
        TestBed.configureTestingModule({
            imports: [UpdateIndicatorComponent],
            providers: [
                { provide: SystemUpdateService, useValue: svc },
                TopbarPanelStore,
            ],
        });
        const fixture = TestBed.createComponent(UpdateIndicatorComponent);
        fixture.detectChanges();
        return { fixture, svc };
    }

    it('renders nothing when the feature is unavailable', () => {
        const { fixture } = setup({ available: false, ready: false, behind: 0 });
        expect(fixture.nativeElement.querySelector('button')).toBeNull();
    });

    it('renders a button when available', () => {
        const { fixture } = setup({ available: true, ready: true, behind: 2 });
        expect(fixture.nativeElement.querySelector('button')).not.toBeNull();
    });

    it('apply() is called from the popover action', () => {
        const { fixture, svc } = setup({ available: true, ready: true, behind: 2 });
        fixture.componentInstance.onApply();
        expect(svc.apply).toHaveBeenCalled();
    });
});
