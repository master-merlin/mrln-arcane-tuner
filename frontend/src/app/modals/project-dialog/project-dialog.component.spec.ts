import { describe, it, expect, vi } from 'vitest';
import { of } from 'rxjs';
import { ProjectDialogComponent } from './project-dialog.component';

interface ConfirmData {
  destructive?: boolean;
  onConfirm?: () => void;
}

function signalStub<T>(value: T): (() => T) & { set: (v: T) => void } {
  return Object.assign(() => value, { set: vi.fn() }) as (() => T) & { set: (v: T) => void };
}

describe('ProjectDialogComponent.deleteProject — themed confirm', () => {
  it('opens the destructive confirm modal and only deletes on confirm', () => {
    const deleteProject = vi.fn().mockReturnValue(of({}));
    const openModal = vi.fn();
    const ctx = {
      isEdit: () => true,
      data: { projectId: 'p1' },
      form: { get: () => ({ value: 'Demo' }) },
      overlay: { openModal },
      submitting: signalStub(false),
      projects: { deleteProject, loadProjects: vi.fn() },
      toast: { success: vi.fn(), error: vi.fn() },
      scope: { projectId: () => null, setGlobal: vi.fn() },
      router: { navigate: vi.fn() },
      close: vi.fn(),
    };

    (ProjectDialogComponent.prototype as unknown as Record<string, (...a: unknown[]) => unknown>)['deleteProject']
      .call(ctx);

    expect(openModal).toHaveBeenCalledWith('confirm', expect.objectContaining({ destructive: true }));
    expect(deleteProject).not.toHaveBeenCalled();

    const data = openModal.mock.calls.at(-1)![1] as ConfirmData;
    data.onConfirm!();
    expect(deleteProject).toHaveBeenCalledWith('p1');
  });
});
