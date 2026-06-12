import type { Mock } from 'vitest';
import { TestBed } from '@angular/core/testing';
import { of, Subject } from 'rxjs';

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
 * Editing while a system default is active must create ONE user copy derived
 * from the live form ("Default by User") and keep saving into it — the legacy
 * behavior. The old code created a brand-new row on every edit-burst/session
 * (no reuse, no in-flight guard, never persisted as active), so identically
 * named "Default by User" rows accumulated and a reload silently landed back
 * on Default — the next edit then spawned yet another copy that looked like
 * it was "derived from some other user template with the same name".
 */
describe('TrainingTemplateSelectorComponent — auto-save copy-on-edit of defaults', () => {
    let svc: {
        listTrainingTemplates: Mock;
        createTrainingTemplate: Mock;
        updateTemplate: Mock;
    };
    let projects: { getPreferences: Mock; updatePreferences: Mock };

    function build(templates: Template[], prefs: Record<string, unknown> = {}) {
        svc = {
            listTrainingTemplates: vi.fn().mockReturnValue(of(templates)),
            createTrainingTemplate: vi.fn().mockImplementation((data: Partial<Template>) =>
                of(tpl({ ...data, id: 'new-id', project_id: data.project_id ?? null } as Partial<Template>))),
            updateTemplate: vi.fn().mockImplementation((_d: string, id: string, data: Partial<Template>) =>
                of(tpl({ ...templates.find(t => t.id === id), ...data, id } as Partial<Template>))),
        };
        projects = {
            getPreferences: vi.fn().mockReturnValue(of({ training_selections: prefs })),
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
        fixture.detectChanges();
        const c = fixture.componentInstance;
        c.suppressAutoSave.set(false); // load's auto-apply leaves it suppressed
        return c;
    }

    it('creates "Default by User" on first edit from the virtual Default and persists it active', () => {
        const c = build([]);
        expect(c.activeTemplateId()).toBe('default');

        c.triggerAutoSave({ learning_rate: 1 }, 'd1');

        expect(svc.createTrainingTemplate).toHaveBeenCalledTimes(1);
        expect(svc.createTrainingTemplate.mock.calls[0][0]).toMatchObject({
            name: 'Default by User', definition_id: 'd1', config: { learning_rate: 1 },
        });
        expect(c.activeTemplateId()).toBe('new-id');
        // Persisted so a reload returns to the copy instead of Default.
        expect(projects.updatePreferences).toHaveBeenCalledTimes(1);
        const [, updates] = projects.updatePreferences.mock.calls[0];
        expect((updates.training_selections as Record<string, unknown>)['active_training_template']).toBe('new-id');
    });

    it('reuses an existing "Default by User" (same definition) instead of creating a duplicate', () => {
        const existing = tpl({ id: 'dbu1', name: 'Default by User', definition_id: 'd1' });
        const c = build([existing]);
        c.activeTemplateId.set('default');

        c.triggerAutoSave({ learning_rate: 2 }, 'd1');

        expect(svc.createTrainingTemplate).not.toHaveBeenCalled();
        expect(svc.updateTemplate).toHaveBeenCalledTimes(1);
        expect(svc.updateTemplate.mock.calls[0][1]).toBe('dbu1');
        expect(svc.updateTemplate.mock.calls[0][2]).toMatchObject({ config: { learning_rate: 2 } });
        expect(c.activeTemplateId()).toBe('dbu1');
    });

    it('creates a separate copy for a different definition (no cross-definition reuse)', () => {
        const existing = tpl({ id: 'dbu1', name: 'Default by User', definition_id: 'd1' });
        const c = build([existing]);
        c.activeTemplateId.set('default');

        c.triggerAutoSave({ learning_rate: 3 }, 'd2');

        expect(svc.updateTemplate).not.toHaveBeenCalled();
        expect(svc.createTrainingTemplate).toHaveBeenCalledTimes(1);
        expect(svc.createTrainingTemplate.mock.calls[0][0]).toMatchObject({ definition_id: 'd2' });
    });

    it('never updates an is_default template in place (even when not readonly)', () => {
        const sysDefault = tpl({ id: 'sys1', name: 'Default', is_default: true, readonly: false });
        const c = build([sysDefault]);
        c.activeTemplateId.set('sys1');

        c.triggerAutoSave({ learning_rate: 4 }, 'd1');

        // Must NOT write through to the system default…
        expect(svc.updateTemplate).not.toHaveBeenCalledWith('training', 'sys1', expect.anything());
        // …it spawns the user copy instead.
        expect(svc.createTrainingTemplate).toHaveBeenCalledTimes(1);
    });

    it('does not create a second copy while the first create is still in flight', () => {
        const create$ = new Subject<Template>();
        const c = build([]);
        svc.createTrainingTemplate.mockReturnValue(create$.asObservable());
        c.activeTemplateId.set('default');

        c.triggerAutoSave({ a: 1 }, 'd1');
        c.triggerAutoSave({ a: 2 }, 'd1'); // debounced burst lands before the POST resolves

        expect(svc.createTrainingTemplate).toHaveBeenCalledTimes(1);

        create$.next(tpl({ id: 'new-id', name: 'Default by User', definition_id: 'd1' }));
        create$.complete();

        // The buffered second edit is flushed into the created copy.
        expect(svc.updateTemplate).toHaveBeenCalledTimes(1);
        expect(svc.updateTemplate.mock.calls[0][1]).toBe('new-id');
        expect(svc.updateTemplate.mock.calls[0][2]).toMatchObject({ config: { a: 2 } });
    });

    it('keeps plain auto-save semantics for an editable active template', () => {
        const mine = tpl({ id: 'mine', name: 'My Template' });
        const c = build([mine]);
        c.activeTemplateId.set('mine');

        c.triggerAutoSave({ b: 1 }, 'd1');

        expect(svc.createTrainingTemplate).not.toHaveBeenCalled();
        expect(svc.updateTemplate).toHaveBeenCalledWith('training', 'mine',
            expect.objectContaining({ config: { b: 1 } }));
    });
});
