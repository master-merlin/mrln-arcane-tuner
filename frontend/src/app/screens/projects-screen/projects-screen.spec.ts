import { describe, it, expect, vi } from 'vitest';
import { of } from 'rxjs';
import { ProjectsScreen } from './projects-screen';

interface ConfirmData {
  destructive?: boolean;
  onConfirm?: () => void;
}

function invoke(method: string, ctx: Record<string, unknown>, ...args: unknown[]): unknown {
  const proto = ProjectsScreen.prototype as unknown as Record<string, (...a: unknown[]) => unknown>;
  return proto[method].apply(ctx, args);
}

describe('ProjectsScreen.deleteProject — themed confirm', () => {
  it('opens the destructive confirm modal and does not delete until confirmed', () => {
    const deleteProject = vi.fn().mockReturnValue(of({}));
    const openModal = vi.fn();
    const ctx = {
      overlay: { openModal },
      projects: { deleteProject, loadProjects: vi.fn() },
      toast: { success: vi.fn(), error: vi.fn() },
      afterDeleteProject: vi.fn(),
    };

    invoke('deleteProject', ctx, { id: 'p1', name: 'Demo' }, { stopPropagation: vi.fn() });

    expect(openModal).toHaveBeenCalledWith('confirm', expect.objectContaining({ destructive: true }));
    expect(deleteProject).not.toHaveBeenCalled();

    const data = openModal.mock.calls.at(-1)![1] as ConfirmData;
    data.onConfirm!();
    expect(deleteProject).toHaveBeenCalledWith('p1');
  });
});
