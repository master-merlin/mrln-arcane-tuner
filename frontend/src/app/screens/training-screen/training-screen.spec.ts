import { Component, input, output, signal } from '@angular/core';
import { TestBed } from '@angular/core/testing';
import { of, Subject } from 'rxjs';

import { TrainingScreen } from './training-screen';
import { TrainingDynamicConfigComponent } from '../../components/training/training-dynamic-config/training-dynamic-config';
import { TrainingToc } from '../../components/training/training-toc/training-toc';
import { TrainingEstimateRail } from '../../components/training/training-estimate-rail/training-estimate-rail';
import { EstimateWallComponent } from '../../components/training/estimate-wall/estimate-wall';
import { JobService } from '../../services/job';
import { ModelService } from '../../services/model.service';
import { ToastService } from '../../services/toast';
import { DatasetStore } from '../../state/dataset.store';
import { ScopeStore } from '../../state/scope.store';
import { TrainingHandoffService } from '../../state/training-handoff.service';
import type { SchemaNode } from '../../components/training/schema-node';
import type { ModelDefinition } from '../../services/model.service';

/**
 * T12 — the Training screen shows a section-shaped form skeleton (not a bare
 * "Loading…" line) while the model list + plugin schema are in flight, then
 * swaps to the real dynamic-config form once the schema resolves. A `loading`
 * signal is the single source of truth for which side of that gate renders.
 *
 * The four heavy child components are replaced by inert stubs sharing their
 * selectors (via a parent-imports swap) so instantiating the real form — which
 * fires HTTP on construction — never happens in this shell-only spec.
 */
const SCHEMA = { type: 'object', properties: {} } as unknown as SchemaNode;
const DEF = { id: 'flux', family: 'flux', name: 'Flux' } as unknown as ModelDefinition;

@Component({ selector: 'app-training-dynamic-config', standalone: true, template: '' })
class StubDynamicConfig {
  schema = input<SchemaNode | undefined>();
  availableModels = input<ModelDefinition[]>([]);
  projectId = input<string | null>(null);
  segmentsChanged = output<unknown>();
  estimateChanged = output<unknown>();
  configSubmitted = output<unknown>();
}
@Component({ selector: 'app-training-toc', standalone: true, template: '' })
class StubToc {
  segments = input<unknown[]>([]);
  activeId = input<string | null>(null);
  jump = output<string>();
}
@Component({ selector: 'app-estimate-wall', standalone: true, template: '' })
class StubEstimateWall {
  estimate = input<unknown>(null);
  ready = input(false);
  recomputing = input(false);
  emptyText = input('');
  updateStats = output<void>();
}
@Component({ selector: 'app-training-estimate-rail', standalone: true, template: '' })
class StubEstimateRail {
  report = input<unknown>(null);
}

describe('TrainingScreen loading skeleton (T12)', () => {
  let schema$: Subject<SchemaNode>;

  function build() {
    schema$ = new Subject<SchemaNode>();
    TestBed.configureTestingModule({
      providers: [
        {
          provide: JobService,
          useValue: {
            getPluginSchema: () => schema$.asObservable(),
            recomputeStats: () => of({}),
            createJob: () => of({}),
            estimate: () => of(null),
          },
        },
        { provide: ModelService, useValue: { getDefinitions: () => of([DEF]) } },
        { provide: ToastService, useValue: { error: () => {}, success: () => {}, warning: () => {} } },
        { provide: DatasetStore, useValue: { loadAll: () => Promise.resolve() } },
        { provide: ScopeStore, useValue: { projectId: () => null } },
        { provide: TrainingHandoffService, useValue: { pending: signal(null), consume: () => null } },
      ],
    });
    // Swap the heavy children for inert same-selector stubs so the real form
    // (which does HTTP on construction) never instantiates.
    TestBed.overrideComponent(TrainingScreen, {
      remove: { imports: [TrainingDynamicConfigComponent, TrainingToc, TrainingEstimateRail, EstimateWallComponent] },
      add: { imports: [StubDynamicConfig, StubToc, StubEstimateWall, StubEstimateRail] },
    });
    const fixture = TestBed.createComponent(TrainingScreen);
    return { fixture, comp: fixture.componentInstance as any };
  }

  it('is loading while the schema fetch is in flight and not after it resolves', () => {
    const { fixture, comp } = build();
    fixture.detectChanges(); // models resolve → schema fetch subscribed (pending)
    expect(comp.loading()).toBe(true);

    schema$.next(SCHEMA);
    fixture.detectChanges();
    expect(comp.loading()).toBe(false);
  });

  it('clears loading when the schema fetch errors (no infinite skeleton)', () => {
    const { fixture, comp } = build();
    fixture.detectChanges();
    expect(comp.loading()).toBe(true);

    schema$.error({ message: 'boom' });
    fixture.detectChanges();
    expect(comp.loading()).toBe(false);
  });

  it('renders the form skeleton while loading and the real form once loaded', () => {
    const { fixture } = build();
    fixture.detectChanges();
    const el: HTMLElement = fixture.nativeElement;

    expect(el.querySelector('[data-testid="training-skeleton"]')).toBeTruthy();
    expect(el.querySelector('app-training-dynamic-config')).toBeFalsy();

    schema$.next(SCHEMA);
    fixture.detectChanges();

    expect(el.querySelector('[data-testid="training-skeleton"]')).toBeFalsy();
    expect(el.querySelector('app-training-dynamic-config')).toBeTruthy();
  });
});
