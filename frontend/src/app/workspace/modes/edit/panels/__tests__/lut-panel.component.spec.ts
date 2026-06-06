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
    exportCube = jasmine.createSpy('exportCube');
}
class StubToastService {
    success = jasmine.createSpy('success');
    error   = jasmine.createSpy('error');
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
        datasetSvc.exportCube.and.returnValue(of(blob));
        const clickSpy = jasmine.createSpy('click');
        spyOn(document, 'createElement').and.returnValue({
            href: '', download: '', click: clickSpy,
        } as unknown as HTMLAnchorElement);
        spyOn(URL, 'createObjectURL').and.returnValue('blob://stub');
        spyOn(URL, 'revokeObjectURL');

        component.exportStack();

        expect(datasetSvc.exportCube).toHaveBeenCalledOnceWith('My DS', state.curves().params);
        expect(URL.createObjectURL).toHaveBeenCalledOnceWith(blob);
        expect(clickSpy).toHaveBeenCalledOnceWith();
        expect(URL.revokeObjectURL).toHaveBeenCalledOnceWith('blob://stub');
        expect(toast.success).toHaveBeenCalledOnceWith('CUBE file exported');
        expect(toast.error).not.toHaveBeenCalled();
    });

    it('toasts an error message when exportCube fails', () => {
        datasetSvc.exportCube.and.returnValue(throwError(() => new Error('boom')));

        component.exportStack();

        expect(toast.error).toHaveBeenCalledOnceWith('Failed to export CUBE file');
        expect(toast.success).not.toHaveBeenCalled();
    });

    it('toasts an error and skips the request when no dataset is open', () => {
        state.datasetName.set('');

        component.exportStack();

        expect(datasetSvc.exportCube).not.toHaveBeenCalled();
        expect(toast.error).toHaveBeenCalledOnceWith('Open an image before exporting a LUT');
        expect(toast.success).not.toHaveBeenCalled();
    });
});
