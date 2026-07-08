import { describe, it, expect, vi } from 'vitest';
import { of, throwError } from 'rxjs';
import { persistJobConfig } from './job-config-save';
import type { JobService } from '../services/job';
import type { ToastService } from '../services/toast';

function deps() {
    const updateJobConfig = vi.fn();
    const jobs = { updateJobConfig } as unknown as JobService;
    const toast = { success: vi.fn(), error: vi.fn() } as unknown as ToastService;
    return { jobs, toast, updateJobConfig };
}

describe('persistJobConfig — shared job-config save path', () => {
    it('rejects invalid JSON without calling the service or side effects', () => {
        const { jobs, toast, updateJobConfig } = deps();
        const onSuccess = vi.fn();
        const onSettled = vi.fn();

        const started = persistJobConfig(jobs, toast, 'j1', '{ not json', { onSuccess, onSettled });

        expect(started).toBe(false);
        expect(updateJobConfig).not.toHaveBeenCalled();
        expect(toast.error).toHaveBeenCalledWith('Invalid JSON — cannot save.');
        expect(onSuccess).not.toHaveBeenCalled();
        expect(onSettled).not.toHaveBeenCalled();
    });

    it('parses valid JSON, persists, then fires success + settled', () => {
        const { jobs, toast, updateJobConfig } = deps();
        (updateJobConfig as ReturnType<typeof vi.fn>).mockReturnValue(of({}));
        const onSuccess = vi.fn();
        const onSettled = vi.fn();

        const started = persistJobConfig(jobs, toast, 'j1', '{"a":1}', { onSuccess, onSettled });

        expect(started).toBe(true);
        expect(updateJobConfig).toHaveBeenCalledWith('j1', { a: 1 });
        expect(toast.success).toHaveBeenCalledWith('Job config saved.');
        expect(onSuccess).toHaveBeenCalledTimes(1);
        expect(onSettled).toHaveBeenCalledTimes(1);
    });

    it('on server error toasts the detail and settles without success', () => {
        const { jobs, toast, updateJobConfig } = deps();
        (updateJobConfig as ReturnType<typeof vi.fn>).mockReturnValue(
            throwError(() => ({ error: { detail: 'nope' } })),
        );
        const onSuccess = vi.fn();
        const onSettled = vi.fn();

        persistJobConfig(jobs, toast, 'j1', '{}', { onSuccess, onSettled });

        expect(toast.error).toHaveBeenCalledWith('Save failed: nope');
        expect(onSuccess).not.toHaveBeenCalled();
        expect(onSettled).toHaveBeenCalledTimes(1);
    });
});
