import type { Mock } from 'vitest';
import { TestBed } from '@angular/core/testing';
import { of } from 'rxjs';

import { TrainingTemplateSelectorComponent } from './training-template-selector';
import { TemplateService, Template } from '../../../services/template.service';
import { ProjectService } from '../../../services/project.service';
import { ToastService } from '../../../services/toast';

function tpl(over: Partial<Template>): Template {
    return {
        id: 't', name: 'T', project_id: null, config: {}, created_at: 0, updated_at: 0,
        used_count: 0, is_default: false, readonly: false, definition_id: 'd1', ...over,
    };
}

/**
 * On entry the Training screen auto-selects a template but never *applies* it,
 * so the estimate wall reflects bare form defaults instead of the selected
 * template. The selector must emit `templateApplied` once on load for whatever
 * template it shows as active — the same path a manual selection takes — tagged
 * `auto: true` so the parent can yield to a Jobs-screen handoff.
 */
describe('TrainingTemplateSelectorComponent — auto-apply active template on load', () => {
    let svc: {
        listTrainingTemplates: Mock;
        recordUse: Mock;
    };
    let projects: {
        getPreferences: Mock;
        updatePreferences: Mock;
    };
    let emitted: Array<{
        config: unknown;
        isDefault: boolean;
        definitionId?: string;
        auto?: boolean;
    }>;

    function build(templates: Template[], trainingSelections: Record<string, unknown> = {}) {
        svc = {
            listTrainingTemplates: vi.fn().mockReturnValue(of(templates)),
            recordUse: vi.fn(),
        };
        projects = {
            getPreferences: vi.fn().mockReturnValue(of({ training_selections: trainingSelections })),
            updatePreferences: vi.fn().mockReturnValue(of({})),
        };
        TestBed.configureTestingModule({
            imports: [TrainingTemplateSelectorComponent],
            providers: [
                { provide: TemplateService, useValue: svc },
                { provide: ProjectService, useValue: projects },
                { provide: ToastService, useValue: {} },
            ],
        });
        const fixture = TestBed.createComponent(TrainingTemplateSelectorComponent);
        emitted = [];
        fixture.componentInstance.templateApplied.subscribe((e) => emitted.push(e));
        fixture.detectChanges(); // runs ngOnInit + effects → loadTrainingSettings (sync of())
        return fixture;
    }

    it('applies the synthetic Default once when no real default exists', () => {
        build([tpl({ id: 'custom1', name: 'Custom', is_default: false })]);
        expect(emitted.length).toBe(1);
        expect(emitted[0].isDefault).toBe(true);
        expect(emitted[0].auto).toBe(true);
    });

    it('applies a real default/first template (no synthetic) on load', () => {
        build([tpl({ id: 'glob', name: 'Global', is_default: true, config: { learning_rate: 5e-6 } })]);
        expect(emitted.length).toBe(1);
        expect(emitted[0].auto).toBe(true);
        // Resolves to a real, selectable option (the active id 'default' matches none).
        expect(emitted[0].config).toEqual({ learning_rate: 5e-6 });
    });

    it('does not re-apply on a second load for the same project (emits once)', () => {
        const fixture = build([tpl({ id: 'custom1', name: 'Custom' })]);
        fixture.componentInstance.loadTrainingSettings();
        fixture.detectChanges();
        expect(emitted.length).toBe(1);
    });

    it('still emits a manual (non-auto) selection from the dropdown', () => {
        const fixture = build([tpl({ id: 'custom1', name: 'Custom', config: { a: 1 } })]);
        emitted.length = 0;
        fixture.componentInstance.applyTemplate('custom1');
        expect(emitted.length).toBe(1);
        expect(emitted[0].auto).toBeFalsy();
    });

    it('records a use on a manual selection but never on the auto-apply', () => {
        // The load-time apply already ran inside build() and is tagged auto —
        // restoring a selection is not choosing one, so it must not tick.
        const fixture = build(
            [tpl({ id: 'a', name: 'Phase I', is_default: true }), tpl({ id: 'b', name: 'Phase II' })],
        );
        expect(svc.recordUse).not.toHaveBeenCalled();

        fixture.componentInstance.applyTemplate('b');
        expect(svc.recordUse).toHaveBeenCalledWith('training', 'b');

        // Re-picking what is already active is not a new use.
        fixture.componentInstance.applyTemplate('b');
        expect(svc.recordUse).toHaveBeenCalledTimes(1);
    });

    it('never records a use for the synthetic Default entry', () => {
        // 'default' is a UI placeholder with no row behind it — a tick would
        // POST /templates/training/default/use against a template that does
        // not exist.
        const fixture = build([tpl({ id: 'custom1', name: 'Custom' })]);
        svc.recordUse.mockClear();
        fixture.componentInstance.applyTemplate('default');
        expect(svc.recordUse).not.toHaveBeenCalled();
    });

    it('restores the persisted active template on load instead of the first one', () => {
        // Two templates; preferences point at the SECOND — load must apply it,
        // not templates[0]. (Mirrors a reload returning to the edited template.)
        build(
            [tpl({ id: 'a', name: 'Phase I', config: { x: 1 } }),
             tpl({ id: 'b', name: 'Phase I', config: { x: 2 } })],
            { active_training_template: 'b' },
        );
        expect(emitted.length).toBe(1);
        expect(emitted[0].auto).toBe(true);
        expect(emitted[0].config).toEqual({ x: 2 });
    });

    it('falls back to the first template when the persisted id no longer exists', () => {
        // `is_default` so no synthetic Default is prepended → templates[0] is 'a'.
        build(
            [tpl({ id: 'a', name: 'Phase I', is_default: true, config: { x: 1 } })],
            { active_training_template: 'deleted-id' },
        );
        expect(emitted[0].config).toEqual({ x: 1 });
    });

    it('persists the active template id on a manual selection (merging training_selections)', () => {
        const fixture = build(
            [tpl({ id: 'a', name: 'Phase I' }), tpl({ id: 'b', name: 'Phase I' })],
            { saved_masking_concepts: ['x'] }, // pre-existing key must be preserved
        );
        projects.updatePreferences.mockClear();
        fixture.componentInstance.applyTemplate('b'); // manual (auto defaults false)
        expect(projects.updatePreferences).toHaveBeenCalledTimes(1);
        const [, updates] = projects.updatePreferences.mock.calls[0];
        expect(updates.training_selections).toEqual({
            saved_masking_concepts: ['x'],
            active_training_template: 'b',
        });
    });

    it('does NOT persist on the auto-apply (restoring, not choosing)', () => {
        build([tpl({ id: 'a', name: 'Phase I' })], { active_training_template: 'a' });
        // Only getPreferences (the load) ran; no write-back from the auto-apply.
        expect(projects.updatePreferences).not.toHaveBeenCalled();
    });
});
