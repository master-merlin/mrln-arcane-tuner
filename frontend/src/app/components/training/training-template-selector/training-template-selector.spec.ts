import type { Mock } from 'vitest';
import { TestBed } from '@angular/core/testing';
import { of } from 'rxjs';

import { TrainingTemplateSelectorComponent } from './training-template-selector';
import { TemplateService, Template } from '../../../services/template.service';
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
    };
    let emitted: Array<{
        config: unknown;
        isDefault: boolean;
        definitionId?: string;
        auto?: boolean;
    }>;

    function build(templates: Template[]) {
        svc = { listTrainingTemplates: vi.fn().mockReturnValue(of(templates)) };
        TestBed.configureTestingModule({
            imports: [TrainingTemplateSelectorComponent],
            providers: [
                { provide: TemplateService, useValue: svc },
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
});
