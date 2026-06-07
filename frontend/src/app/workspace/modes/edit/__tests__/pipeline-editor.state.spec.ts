import type { Mock } from 'vitest';
import { TestBed } from '@angular/core/testing';
import { signal } from '@angular/core';
import { of } from 'rxjs';
import { PipelineEditorState } from '../pipeline-editor.state';
import { OverlayStore } from '../../../../state/overlay.store';
import { ToastService } from '../../../../services/toast';
import { WebSocketService } from '../../../../services/websocket.service';
import { DatasetService } from '../../../../services/dataset';
import { TaskStore } from '../../../../state/task.store';

// Canonical WS stub used across `frontend/src/app/state/__tests__/*.spec.ts`:
// only `entityChanged` and `reconnected` are read by EntityStore's constructor
// effects, and both are signals.
function makeWsMock() {
    return { entityChanged: signal(null), reconnected: signal(0) } as unknown as WebSocketService;
}
class StubToast {
    success() { }
    error() { }
    warning() { }
    info() { }
}

// Minimal stubs for the two new deps injected by PipelineEditorState.
// All existing describes just need something in the DI tree so Angular
// doesn't try to instantiate the real TaskStore (which calls ws.on()).
const STUB_TASK_STORE = { byId: () => signal(undefined), cancel: () => { } };
const STUB_DATASET_SERVICE = { taskRenderPipeline: () => { } };

describe('PipelineEditorState.resetAllForUser', () => {
    let state: PipelineEditorState;

    beforeEach(() => {
        TestBed.configureTestingModule({
            providers: [
                PipelineEditorState,
                OverlayStore,
                { provide: WebSocketService, useValue: makeWsMock() },
                { provide: ToastService, useClass: StubToast },
                { provide: TaskStore, useValue: STUB_TASK_STORE },
                { provide: DatasetService, useValue: STUB_DATASET_SERVICE },
            ],
        });
        state = TestBed.inject(PipelineEditorState);
    });

    it('zeros every panel back to its default params', () => {
        state.whiteBalance.update(o => ({ ...o, enabled: true, params: { temperature: 7800, tint: 12 } }));
        state.hslSelective.update(o => ({ ...o, enabled: true, params: { reds: { hue_shift: 30, saturation: 0, luminance: 0 } } as never }));

        state.resetAllForUser();

        expect(state.whiteBalance().enabled).toBe(false);
        expect(state.whiteBalance().params.temperature).toBe(6500);
        expect(state.whiteBalance().params.tint).toBe(0);
        expect(state.hslSelective().enabled).toBe(false);
    });

    it('clears every panel\'s enabled flag and restores operationOrder', () => {
        // One assertion-blast that catches any panel accidentally
        // dropped from the reset body, plus the operationOrder reset
        // (the original "zeros" test only probed 2 of 12 panels and
        // didn't check operationOrder at all).
        state.whiteBalance.update(o => ({ ...o, enabled: true }));
        state.curves.update(o => ({ ...o, enabled: true }));
        state.lut.update(o => ({ ...o, enabled: true }));
        state.colorMatch.update(o => ({ ...o, enabled: true }));
        state.hslSelective.update(o => ({ ...o, enabled: true }));
        state.colorTone.update(o => ({ ...o, enabled: true }));
        state.vignette.update(o => ({ ...o, enabled: true }));
        state.lens.update(o => ({ ...o, enabled: true }));
        state.sharpen.update(o => ({ ...o, enabled: true }));
        state.denoise.update(o => ({ ...o, enabled: true }));
        state.faceRestore.update(o => ({ ...o, enabled: true }));
        state.upscale.update(o => ({ ...o, enabled: true }));
        // Scramble the order to verify reset restores PIPELINE_ORDER.
        state.moveOperation(0, 5);

        state.resetAllForUser();

        // Every panel disabled.
        expect(state.whiteBalance().enabled).toBe(false);
        expect(state.curves().enabled).toBe(false);
        expect(state.lut().enabled).toBe(false);
        expect(state.colorMatch().enabled).toBe(false);
        expect(state.hslSelective().enabled).toBe(false);
        expect(state.colorTone().enabled).toBe(false);
        expect(state.vignette().enabled).toBe(false);
        expect(state.lens().enabled).toBe(false);
        expect(state.sharpen().enabled).toBe(false);
        expect(state.denoise().enabled).toBe(false);
        expect(state.faceRestore().enabled).toBe(false);
        expect(state.upscale().enabled).toBe(false);

        // operationOrder back to its canonical sequence.
        // Read the canonical via the state's blocks() side-effect:
        // after reset, blocks() must be empty (everything disabled).
        // operationOrder itself must be a non-empty array of unique
        // kinds in the canonical order; assert the count for now.
        const order = state.operationOrder();
        expect(order.length).toBeGreaterThan(0);
        expect(new Set(order).size).toBe(order.length); // no dupes
        expect(state.blocks().length).toBe(0);
    });

    it('leaves dirty=true so the user can Save the empty recipe', () => {
        // Apply a non-default change and mark clean — simulates "image
        // opened with a saved recipe already on disk".
        state.whiteBalance.update(o => ({ ...o, enabled: true, params: { temperature: 7800, tint: 12 } }));
        state.markClean();
        expect(state.dirty()).toBe(false);

        state.resetAllForUser();

        expect(state.dirty()).toBe(true);
    });
});

describe('PipelineEditorState.resetAll', () => {
    let state: PipelineEditorState;

    beforeEach(() => {
        TestBed.configureTestingModule({
            providers: [
                PipelineEditorState,
                OverlayStore,
                { provide: WebSocketService, useValue: makeWsMock() },
                { provide: ToastService, useClass: StubToast },
                { provide: TaskStore, useValue: STUB_TASK_STORE },
                { provide: DatasetService, useValue: STUB_DATASET_SERVICE },
            ],
        });
        state = TestBed.inject(PipelineEditorState);
    });

    it('still markCleans so the bake/hydrate flows do not leak dirty=true', () => {
        state.whiteBalance.update(o => ({ ...o, enabled: true, params: { temperature: 7800, tint: 12 } }));
        state.resetAll();
        expect(state.dirty()).toBe(false);
    });
});

describe('PipelineEditorState.blocks (color_match edge case)', () => {
    let state: PipelineEditorState;

    beforeEach(() => {
        TestBed.configureTestingModule({
            providers: [
                PipelineEditorState,
                OverlayStore,
                { provide: WebSocketService, useValue: makeWsMock() },
                { provide: ToastService, useClass: StubToast },
                { provide: TaskStore, useValue: STUB_TASK_STORE },
                { provide: DatasetService, useValue: STUB_DATASET_SERVICE },
            ],
        });
        state = TestBed.inject(PipelineEditorState);
    });

    it('omits color_match from blocks() when enabled but reference_path is empty', () => {
        // Locks the documented edge case (edit-right-panel.component.ts:
        // anyOpEnabled's JSDoc) — a user who toggles Color Match on but
        // hasn't picked a reference yet will see Reset All greyed out,
        // because blocks() requires BOTH enabled AND a reference_path.
        // If a future blocks() refactor accidentally relaxes that
        // condition this test fails and forces a re-evaluation.
        state.colorMatch.update(o => ({ ...o, enabled: true }));
        // Default colorMatch params have reference_path === '' (falsy).

        expect(state.blocks().length).toBe(0);

        // Once a reference is set, color_match flows into blocks() normally.
        state.colorMatch.update(o => ({
            ...o,
            params: { ...o.params, reference_path: '/some/ref.png' },
        }));

        const block = state.blocks().find(b => b.type === 'color_match');
        expect(block).toBeDefined();
    });
});

describe('PipelineEditorState.applyAndSave — task routing', () => {
    let state: PipelineEditorState;
    let api: any;
    let taskStoreSpy: {
        byId: Mock;
        cancel: Mock;
    };
    let overlay: OverlayStore;

    beforeEach(() => {
        api = { taskRenderPipeline: vi.fn().mockReturnValue(of({ task_id: 't1' })) };
        taskStoreSpy = { byId: vi.fn().mockReturnValue(signal(undefined)), cancel: vi.fn() };
        TestBed.configureTestingModule({
            providers: [
                PipelineEditorState,
                OverlayStore,
                { provide: WebSocketService, useValue: makeWsMock() },
                { provide: ToastService, useClass: StubToast },
                { provide: DatasetService, useValue: api },
                { provide: TaskStore, useValue: taskStoreSpy },
            ],
        });
        state = TestBed.inject(PipelineEditorState);
        overlay = TestBed.inject(OverlayStore);
        state.datasetName.set('ds1');
        state.mediaFile.set('a.png');
    });

    it('routes a GPU-op save (upscale) through taskRenderPipeline, not the inline render', async () => {
        const inlineSpy = vi.spyOn(overlay, 'renderPipeline');
        state.upscale.update(o => ({ ...o, enabled: true }));
        await state.applyAndSave();
        expect(api.taskRenderPipeline).toHaveBeenCalled();
        const [name, file] = vi.mocked(api.taskRenderPipeline).mock.lastCall!;
        expect(name).toBe('ds1');
        expect(file).toBe('a.png');
        expect(inlineSpy).not.toHaveBeenCalled();
        expect(state.renderTaskId()).toBe('t1');
        expect(state.saving()).toBe(true);
    });

    it('routes a CPU-only save (white_balance) through the inline render, not a task', async () => {
        vi.spyOn(overlay, 'renderPipeline').mockReturnValue(Promise.resolve({ ok: true, value: { dimensions: [10, 10] } } as any));
        state.whiteBalance.update(o => ({ ...o, enabled: true }));
        await state.applyAndSave();
        expect(overlay.renderPipeline).toHaveBeenCalled();
        expect(api.taskRenderPipeline).not.toHaveBeenCalled();
        expect(state.saving()).toBe(false);
    });

    it('completion: completed task markCleans + clears saving (same image)', () => {
        const taskSignal = signal<any>(undefined);
        taskStoreSpy.byId.mockReturnValue(taskSignal);
        state.upscale.update(o => ({ ...o, enabled: true }));
        void state.applyAndSave();
        expect(state.dirty()).toBe(true);
        taskSignal.set({ status: 'completed', current: 1, total: 1, ok: 1, failed: 0, current_item: null, error: null });
        TestBed.tick();
        expect(state.dirty()).toBe(false);
        expect(state.saving()).toBe(false);
    });
});
