import { describe, it, expect, vi, beforeEach } from 'vitest';
import { TestBed } from '@angular/core/testing';

import { TemplateAutosaveService } from './template-autosave.service';

function setup() {
  TestBed.configureTestingModule({ providers: [TemplateAutosaveService] });
  const svc = TestBed.inject(TemplateAutosaveService);
  const dispatch = vi.fn();
  let suppressed = false;
  svc.configure({ dispatch, suppressed: () => suppressed });
  return {
    svc,
    dispatch,
    setSuppressed: (v: boolean) => { suppressed = v; },
  };
}

describe('TemplateAutosaveService', () => {
  beforeEach(() => TestBed.resetTestingModule());

  it('debounces a burst of schedules into a single dispatch (1200ms) with the latest value', () => {
    vi.useFakeTimers();
    try {
      const { svc, dispatch } = setup();

      svc.schedule({ a: 1 }, 'd1');
      svc.schedule({ a: 2 }, 'd1');
      svc.schedule({ a: 3 }, 'd1');
      vi.advanceTimersByTime(1300);

      expect(dispatch).toHaveBeenCalledTimes(1);
      expect(dispatch).toHaveBeenCalledWith({ a: 3 }, 'd1');
    } finally {
      vi.useRealTimers();
    }
  });

  /**
   * The suppressAutoSave latch guard — the house gotcha. A change scheduled
   * while suppression is latched must NEVER reach a save (it would silently
   * overwrite a user template); once released, changes save again. RED-first.
   */
  it('suppress → change → release → no save fires for the suppressed window', () => {
    vi.useFakeTimers();
    try {
      const { svc, dispatch, setSuppressed } = setup();

      // Latched: a scheduled change during the window must not dispatch.
      setSuppressed(true);
      svc.schedule({ a: 1 }, 'd1');
      vi.advanceTimersByTime(1300);
      expect(dispatch).not.toHaveBeenCalled();

      // Released: subsequent changes save normally.
      setSuppressed(false);
      svc.schedule({ a: 2 }, 'd1');
      vi.advanceTimersByTime(1300);
      expect(dispatch).toHaveBeenCalledTimes(1);
      expect(dispatch).toHaveBeenCalledWith({ a: 2 }, 'd1');
    } finally {
      vi.useRealTimers();
    }
  });

  it('evaluates the suppression latch at DISPATCH time, not schedule time', () => {
    vi.useFakeTimers();
    try {
      const { svc, dispatch, setSuppressed } = setup();

      // Scheduled while NOT suppressed, but suppression latches before the
      // debounce fires → still dropped (latch wins at dispatch).
      svc.schedule({ a: 9 }, 'd1');
      setSuppressed(true);
      vi.advanceTimersByTime(1300);
      expect(dispatch).not.toHaveBeenCalled();
    } finally {
      vi.useRealTimers();
    }
  });
});
