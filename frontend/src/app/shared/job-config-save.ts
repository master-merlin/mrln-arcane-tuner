import type { JobService } from '../services/job';
import type { ToastService } from '../services/toast';

/** Side-effect callbacks for {@link persistJobConfig}. */
export interface PersistJobConfigCallbacks {
    /** Success-only side effects (close modal / refresh list / reset baseline). */
    onSuccess: () => void;
    /** Runs after the request settles (success OR error) — clear the saving flag. */
    onSettled: () => void;
}

/**
 * Single source of truth for persisting a job's raw-JSON config text.
 *
 * Both the standalone job-config modal and the jobs-screen inline Run Config
 * editor previously carried their own copy of this parse → validate → PUT →
 * toast sequence; they now share this helper so the save semantics (error
 * messages, endpoint, success/error toasts) live in one place. The two
 * call sites differ only in their success side effects (the modal closes +
 * fires `onSaved`; the inline panel re-baselines + reloads the store), which
 * are supplied via {@link PersistJobConfigCallbacks}.
 *
 * Returns `true` if the text parsed and a save request was dispatched, or
 * `false` if the JSON was invalid (nothing persisted; the caller should
 * clear any optimistic "saving" flag it set). On invalid input the error
 * toast is raised here, so callers need not duplicate it.
 */
export function persistJobConfig(
    jobs: JobService,
    toast: ToastService,
    jobId: string,
    text: string,
    cb: PersistJobConfigCallbacks,
): boolean {
    let parsed: Record<string, unknown>;
    try {
        parsed = JSON.parse(text);
    } catch {
        toast.error('Invalid JSON — cannot save.');
        return false;
    }
    jobs.updateJobConfig(jobId, parsed).subscribe({
        next: () => {
            toast.success('Job config saved.');
            cb.onSuccess();
            cb.onSettled();
        },
        error: (err: { error?: { detail?: string }; message?: string }) => {
            toast.error('Save failed: ' + (err?.error?.detail || err?.message));
            cb.onSettled();
        },
    });
    return true;
}
