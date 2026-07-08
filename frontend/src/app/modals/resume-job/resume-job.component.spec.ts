import { describe, it, expect, vi, beforeEach } from 'vitest';
import { TestBed, type ComponentFixture } from '@angular/core/testing';
import { ResumeJobModalComponent, type ResumeJobDialogData } from './resume-job.component';
import { OverlayStore } from '../../state/overlay.store';
import type { JobCheckpointMeta } from '../../services/job';

function ckpt(over: Partial<JobCheckpointMeta>): JobCheckpointMeta {
    return {
        filename: 'lora.safetensors', step: 100, is_final: false,
        size_bytes: 1, created_at: 0, resumable: true, checkpoint_dir: 'checkpoint-000100',
        ...over,
    };
}

function setup(over: Partial<ResumeJobDialogData> = {}): {
    fixture: ComponentFixture<ResumeJobModalComponent>;
    comp: ResumeJobModalComponent;
    onRestart: ReturnType<typeof vi.fn>;
    onContinue: ReturnType<typeof vi.fn>;
    overlay: OverlayStore;
} {
    const onRestart = vi.fn();
    const onContinue = vi.fn();
    const data: ResumeJobDialogData = {
        jobId: 'job-1',
        checkpoints: [
            ckpt({ step: 100, checkpoint_dir: 'checkpoint-000100' }),
            ckpt({ step: 500, checkpoint_dir: 'checkpoint-000500' }),
        ],
        onRestart,
        onContinue,
        ...over,
    };
    TestBed.configureTestingModule({ imports: [ResumeJobModalComponent] });
    const overlay = TestBed.inject(OverlayStore);
    overlay.openModal('resume-job', data);
    const fixture = TestBed.createComponent(ResumeJobModalComponent);
    fixture.detectChanges();
    return { fixture, comp: fixture.componentInstance, onRestart, onContinue, overlay };
}

describe('ResumeJobModalComponent', () => {
    beforeEach(() => TestBed.resetTestingModule());

    it('defaults to continue mode with the latest checkpoint selected', () => {
        const { comp } = setup();
        expect(comp.mode()).toBe('continue');
        // Latest = highest step (checkpoint-000500).
        expect(comp.selectedDir()).toBe('checkpoint-000500');
    });

    it('defaults the wipe checkbox to off so data loss is opt-in', () => {
        const { comp } = setup();
        expect(comp.wipe()).toBe(false);
    });

    it('confirm in continue mode calls onContinue with the selected dir and closes', () => {
        const { comp, onContinue, onRestart, overlay } = setup();
        comp.selectedDir.set('checkpoint-000100');
        comp.confirm();
        expect(onContinue).toHaveBeenCalledWith('checkpoint-000100');
        expect(onRestart).not.toHaveBeenCalled();
        expect(overlay.modalStack().length).toBe(0);
    });

    it('confirm in restart mode calls onRestart with the wipe flag and closes', () => {
        const { comp, onRestart, onContinue, overlay } = setup();
        comp.mode.set('restart');
        comp.wipe.set(false);
        comp.confirm();
        expect(onRestart).toHaveBeenCalledWith(false);
        expect(onContinue).not.toHaveBeenCalled();
        expect(overlay.modalStack().length).toBe(0);
    });
});
