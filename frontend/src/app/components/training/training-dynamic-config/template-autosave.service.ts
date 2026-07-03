import { Injectable, DestroyRef, inject } from '@angular/core';
import { Subject } from 'rxjs';
import { debounceTime } from 'rxjs/operators';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';

import type { TrainingConfig } from '../../../services/job';

/**
 * Owns the debounced template-autosave pipeline extracted from
 * training-dynamic-config (F-ARCH-11). The component previously wired a
 * `form.valueChanges → debounceTime(1200) → triggerAutoSave` subscription
 * inline; that dispatch is now this service's job.
 *
 * IMPORTANT — suppressAutoSave latch semantics: the authoritative latch signal
 * remains `TrainingTemplateSelectorComponent.suppressAutoSave` (it is set from
 * ~6 template apply/adopt call sites and read inside `triggerAutoSave`, and the
 * selector's autosave.spec pins it there). This service GATES the debounced
 * dispatch on a `suppressed()` predicate that mirrors that same latch, so a
 * change scheduled while suppression is latched never reaches a save — the
 * exact semantic the house gotcha protects (a regression silently overwrites
 * user templates). The selector's own internal check is kept as defense in
 * depth; both observe the same latch value at dispatch time, so the outcome is
 * identical.
 *
 * Component-provided (NOT root) so the internal subscription ties to the
 * component's lifetime via the injected `DestroyRef`.
 */
@Injectable()
export class TemplateAutosaveService {
  private destroyRef = inject(DestroyRef);

  private trigger$ = new Subject<{ value: TrainingConfig; defId: string }>();

  /** Fires the actual persistence (the selector's `triggerAutoSave`). */
  private dispatch: (value: TrainingConfig, defId: string) => void = () => {};
  /** Mirror of the selector's `suppressAutoSave()` latch. */
  private suppressed: () => boolean = () => false;

  constructor() {
    this.trigger$.pipe(
      debounceTime(1200),
      takeUntilDestroyed(this.destroyRef),
    ).subscribe(({ value, defId }) => {
      // Suppression latched (template apply/adopt in flight) → drop the save so
      // filling defaults / applying a template never rewrites a user template.
      if (this.suppressed()) return;
      this.dispatch(value, defId);
    });
  }

  /** Wire the persistence dispatch + the suppression-latch mirror. */
  configure(opts: {
    dispatch: (value: TrainingConfig, defId: string) => void;
    suppressed: () => boolean;
  }): void {
    this.dispatch = opts.dispatch;
    this.suppressed = opts.suppressed;
  }

  /** Debounced autosave trigger — fed by the component's `form.valueChanges`. */
  schedule(value: TrainingConfig, defId: string): void {
    this.trigger$.next({ value, defId });
  }
}
