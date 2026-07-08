import { describe, it, expect, vi } from 'vitest';
import { of } from 'rxjs';
import { TemplatesLibraryModalComponent } from './templates-library.component';

interface ConfirmData {
  destructive?: boolean;
  onConfirm?: () => void | Promise<void>;
}

function signalStub<T>(value: T): (() => T) & { set: (v: T) => void } {
  return Object.assign(() => value, { set: vi.fn() }) as (() => T) & { set: (v: T) => void };
}

describe('TemplatesLibraryModalComponent.deleteTemplate — themed confirm', () => {
  it('opens the destructive confirm modal and only deletes on confirm', async () => {
    const deleteTemplate = vi.fn().mockReturnValue(of({}));
    const openModal = vi.fn();
    const ctx = {
      overlay: { openModal },
      templateApi: { deleteTemplate },
      activeDomain: () => 'training',
      busy: signalStub<string | null>(null),
      toast: { success: vi.fn(), error: vi.fn() },
      load: vi.fn().mockResolvedValue(undefined),
    };

    (TemplatesLibraryModalComponent.prototype as unknown as Record<string, (...a: unknown[]) => unknown>)['deleteTemplate']
      .call(ctx, { id: 't1', name: 'Tpl', readonly: false });

    expect(openModal).toHaveBeenCalledWith('confirm', expect.objectContaining({ destructive: true }));
    expect(deleteTemplate).not.toHaveBeenCalled();

    const data = openModal.mock.calls.at(-1)![1] as ConfirmData;
    await data.onConfirm!();
    expect(deleteTemplate).toHaveBeenCalledWith('training', 't1');
  });
});
