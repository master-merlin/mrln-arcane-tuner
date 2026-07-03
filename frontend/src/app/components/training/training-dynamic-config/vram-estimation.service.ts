import { Injectable, DestroyRef, inject, signal } from '@angular/core';
import { Subject } from 'rxjs';
import { debounceTime } from 'rxjs/operators';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';

import { JobService, type TrainingEstimate, type TrainingConfig } from '../../../services/job';
import { VRAMReport } from '../../../services/system.service';

/**
 * Owns the debounced VRAM/training-estimate pipeline extracted from
 * training-dynamic-config (F-ARCH-11). The component was doing too many jobs;
 * this service holds the estimate signals, the debounced trigger subject, and
 * the request-building refresh so the component only wires its form's
 * `valueChanges` into `schedule()` and binds the exposed signals.
 *
 * Component-provided (NOT root) so each dynamic-config instance owns its own
 * estimate state and the internal subscription ties to the component's
 * lifetime via the injected `DestroyRef`.
 *
 * Pipeline shape is preserved 1:1 from the original component so the two-stage
 * debounce timing the rebuild-subs spec pins is unchanged: the component
 * debounces `valueChanges` (800ms) into `schedule()`, and this service's
 * trigger subject debounces again (800ms) before calling `refresh()`.
 */
@Injectable()
export class VramEstimationService {
  private jobService = inject(JobService);
  private destroyRef = inject(DestroyRef);

  /** Live VRAM report (peak/available/fits) feeding the in-form budget card. */
  readonly vramReport = signal<VRAMReport | null>(null);
  /** Full data-calibrated estimate (wall time, throughput, output, disk + VRAM). */
  readonly estimate = signal<TrainingEstimate | null>(null);

  private trigger$ = new Subject<void>();

  /**
   * Supplies the current `{ defId, config }` at refresh time. Set by the
   * component so this service stays form-agnostic; it reads the LATEST form
   * state when the debounce fires (not at schedule time), matching the
   * original `refreshVRAMEstimate` which called `form.getRawValue()`.
   */
  private requestFactory: () => { defId: string; config: TrainingConfig } | null = () => null;

  constructor() {
    this.trigger$.pipe(
      debounceTime(800),
      takeUntilDestroyed(this.destroyRef),
    ).subscribe(() => this.refresh());
  }

  /** Wire the config source (the component's form snapshot). */
  configure(requestFactory: () => { defId: string; config: TrainingConfig } | null): void {
    this.requestFactory = requestFactory;
  }

  /** Debounced re-estimate trigger (mirrors the old `vramEstimate$.next()`). */
  schedule(): void {
    this.trigger$.next();
  }

  /**
   * One call to the full estimator: it returns the calibrated VRAM report
   * (feeding the in-form budget card + the shell's VRAM detail rail) PLUS
   * wall time / throughput / output / disk for the shared estimate wall.
   */
  refresh(): void {
    const req = this.requestFactory();
    if (!req || !req.defId) return;

    this.jobService.estimate(req.defId, req.config).subscribe({
      next: (est) => {
        this.estimate.set(est);
        this.vramReport.set(est?.vram ?? null);
      },
      error: (err) => {
        console.warn('[Estimate] Estimation failed', err);
        this.estimate.set(null);
        this.vramReport.set(null);
      },
    });
  }
}
