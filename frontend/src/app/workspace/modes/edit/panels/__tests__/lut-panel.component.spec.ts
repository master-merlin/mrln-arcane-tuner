import { TestBed } from '@angular/core/testing';
import { signal } from '@angular/core';
import { of, throwError, NEVER } from 'rxjs';
import { LutPanelComponent } from '../lut-panel.component';
import { PipelineEditorState } from '../../pipeline-editor.state';
import { OverlayStore } from '../../../../../state/overlay.store';
import { DatasetService } from '../../../../../services/dataset';
import { ToastService } from '../../../../../services/toast';
import { WebSocketService } from '../../../../../services/websocket.service';

class StubDatasetService {
    exportCube = vi.fn();
}
class StubToastService {
    success = vi.fn();
    error = vi.fn();
}
function makeWsMock() {
    // `on()` is needed because this spec builds a component tree that
    // transitively constructs TaskStore, which subscribes via ws.on().
    return {
        entityChanged: signal(null),
        reconnected: signal(0),
        on: () => NEVER,
    } as unknown as WebSocketService;
}

describe('LutPanelComponent.exportStack', () => {
    let component: LutPanelComponent;
    let datasetSvc: StubDatasetService;
    let toast: StubToastService;
    let state: PipelineEditorState;

    beforeEach(() => {
        TestBed.configureTestingModule({
            providers: [
                PipelineEditorState,
                OverlayStore,
                { provide: WebSocketService, useValue: makeWsMock() },
                { provide: DatasetService, useClass: StubDatasetService },
                { provide: ToastService, useClass: StubToastService },
            ],
        });
        component = TestBed.createComponent(LutPanelComponent).componentInstance;
        datasetSvc = TestBed.inject(DatasetService) as unknown as StubDatasetService;
        toast = TestBed.inject(ToastService) as unknown as StubToastService;
        state = TestBed.inject(PipelineEditorState);
        state.datasetName.set('My DS');
    });

    it('passes the current curves to DatasetService.exportCube and toasts on success', () => {
        const blob = new Blob(['LUT body'], { type: 'application/octet-stream' });
        datasetSvc.exportCube.mockReturnValue(of(blob));
        const clickSpy = vi.fn();
        vi.spyOn(document, 'createElement').mockReturnValue({
            href: '', download: '', click: clickSpy,
        } as unknown as HTMLAnchorElement);
        // LANE-75: assert on the spy handles THIS test created, never on the
        // global property — a mock another file left on `URL` would otherwise
        // be reused by vi.spyOn and carry its call count into this assertion.
        const createSpy = vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob://stub');
        const revokeSpy = vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => undefined);

        component.exportStack();

        expect(datasetSvc.exportCube).toHaveBeenCalledTimes(1);

        expect(datasetSvc.exportCube).toHaveBeenCalledWith('My DS', state.curves().params);
        expect(createSpy).toHaveBeenCalledTimes(1);
        expect(createSpy).toHaveBeenCalledWith(blob);
        expect(clickSpy).toHaveBeenCalledTimes(1);
        expect(clickSpy).toHaveBeenCalledWith();
        expect(revokeSpy).toHaveBeenCalledTimes(1);
        expect(revokeSpy).toHaveBeenCalledWith('blob://stub');
        expect(toast.success).toHaveBeenCalledTimes(1);
        expect(toast.success).toHaveBeenCalledWith('CUBE file exported');
        expect(toast.error).not.toHaveBeenCalled();
    });

    it('toasts an error message when exportCube fails', () => {
        datasetSvc.exportCube.mockReturnValue(throwError(() => new Error('boom')));

        component.exportStack();

        expect(toast.error).toHaveBeenCalledTimes(1);

        expect(toast.error).toHaveBeenCalledWith('Failed to export CUBE file');
        expect(toast.success).not.toHaveBeenCalled();
    });

    it('toasts an error and skips the request when no dataset is open', () => {
        state.datasetName.set('');

        component.exportStack();

        expect(datasetSvc.exportCube).not.toHaveBeenCalled();
        expect(toast.error).toHaveBeenCalledTimes(1);
        expect(toast.error).toHaveBeenCalledWith('Open an image before exporting a LUT');
        expect(toast.success).not.toHaveBeenCalled();
    });
});
