import { describe, it, expect, vi, beforeEach } from 'vitest';
import { TestBed } from '@angular/core/testing';
import { of } from 'rxjs';

import { VramEstimationService } from './vram-estimation.service';
import { JobService, type TrainingEstimate } from '../../../services/job';

function estimate(over: Partial<TrainingEstimate> = {}): TrainingEstimate {
  return {
    vram: { peak_mb: 8000, available_mb: 24000, fits: true } as TrainingEstimate['vram'],
    ...over,
  } as TrainingEstimate;
}

function setup() {
  const estimateFn = vi.fn().mockReturnValue(of(estimate()));
  TestBed.configureTestingModule({
    providers: [
      VramEstimationService,
      { provide: JobService, useValue: { estimate: estimateFn } },
    ],
  });
  const svc = TestBed.inject(VramEstimationService);
  return { svc, estimateFn };
}

describe('VramEstimationService', () => {
  beforeEach(() => TestBed.resetTestingModule());

  it('refresh() builds the request from the configured factory and populates the signals', () => {
    const { svc, estimateFn } = setup();
    svc.configure(() => ({ defId: 'flux', config: { network_rank: 16 } }));

    svc.refresh();

    expect(estimateFn).toHaveBeenCalledWith('flux', { network_rank: 16 });
    expect(svc.estimate()).not.toBeNull();
    expect(svc.vramReport()).toEqual({ peak_mb: 8000, available_mb: 24000, fits: true });
  });

  it('refresh() is a no-op (no HTTP) when the factory yields no definition id', () => {
    const { svc, estimateFn } = setup();
    svc.configure(() => null);
    svc.refresh();
    expect(estimateFn).not.toHaveBeenCalled();
  });

  it('clears the signals when the estimator errors', () => {
    const { svc, estimateFn } = setup();
    estimateFn.mockReturnValue({ subscribe: (o: { error: (e: unknown) => void }) => o.error(new Error('boom')) });
    svc.configure(() => ({ defId: 'flux', config: {} }));

    svc.refresh();

    expect(svc.estimate()).toBeNull();
    expect(svc.vramReport()).toBeNull();
  });

  it('schedule() debounces bursts into a single refresh (800ms) reading the LATEST config', () => {
    vi.useFakeTimers();
    try {
      const { svc, estimateFn } = setup();
      let rank = 1;
      svc.configure(() => ({ defId: 'flux', config: { network_rank: rank } }));
      const refreshSpy = vi.spyOn(svc, 'refresh');

      svc.schedule();
      rank = 2;
      svc.schedule();
      rank = 3;
      svc.schedule();
      vi.advanceTimersByTime(900);

      // One coalesced refresh, and it read the newest config value.
      expect(refreshSpy).toHaveBeenCalledTimes(1);
      expect(estimateFn).toHaveBeenCalledTimes(1);
      expect(estimateFn).toHaveBeenCalledWith('flux', { network_rank: 3 });
    } finally {
      vi.useRealTimers();
    }
  });
});
