import type { Mock } from 'vitest';
import { TestBed } from '@angular/core/testing';
import { of, Subject } from 'rxjs';

import { DatasetMaskingSettingsComponent } from './dataset-masking-settings';
import { DatasetService } from '../../../services/dataset';
import { ProjectService } from '../../../services/project.service';
import { TemplateService, Template } from '../../../services/template.service';

function tpl(over: Partial<Template>): Template {
    return {
        id: 't', name: 'T', project_id: null, config: {}, created_at: 0, updated_at: 0,
        used_count: 0, is_default: false, readonly: false, model_id: 'sam3', ...over,
    };
}

/**
 * Editing a system-default masking template must save into ONE user copy
 * ("Custom Settings") derived from the default — never write through to the
 * default, never silently drop the edit, never stack duplicate copies. The
 * old code: only checked `readonly` (an `is_default` row could be edited in
 * place), created a brand-new "Custom Settings" per session (no reuse), and
 * RETURNED early when a model had no template at all (RemBG had no seeded
 * default) — those edits vanished.
 */
describe('DatasetMaskingSettings — copy-on-edit of default templates', () => {
    let svc: {
        listMaskingTemplates: Mock;
        createMaskingTemplate: Mock;
        updateTemplate: Mock;
    };

    function build(templates: Template[], prefs: Record<string, unknown> = {}) {
        svc = {
            listMaskingTemplates: vi.fn().mockReturnValue(of(templates)),
            createMaskingTemplate: vi.fn().mockImplementation((data: Partial<Template>) =>
                of(tpl({ ...data, id: 'new-id' } as Partial<Template>))),
            updateTemplate: vi.fn().mockImplementation((_d: string, id: string, data: Partial<Template>) =>
                of(tpl({ ...templates.find(t => t.id === id), ...data, id } as Partial<Template>))),
        };
        TestBed.configureTestingModule({
            imports: [DatasetMaskingSettingsComponent],
            providers: [
                { provide: DatasetService, useValue: {} },
                {
                    provide: ProjectService, useValue: {
                        activeDatasetProject: () => 'p1',
                        getPreferences: vi.fn().mockReturnValue(of(prefs)),
                        updatePreferences: vi.fn().mockReturnValue(of({})),
                    },
                },
                { provide: TemplateService, useValue: svc },
            ],
        });
        TestBed.overrideComponent(DatasetMaskingSettingsComponent, { set: { template: '' } });
        const fixture = TestBed.createComponent(DatasetMaskingSettingsComponent);
        fixture.detectChanges(); // effects → loadPreferencesAndTemplates (sync of())
        return fixture.componentInstance;
    }

    // The save path is timer-only (setTimeout save debounce); every observable
    // here is a synchronous of()/Subject. Vitest fake timers replace fakeAsync's
    // tick()/flush(): vi.runAllTimers() drains all pending timers like flush()
    // did. 'Date' MUST be faked with the timer fns — RxJS debounceTime compares
    // Date.now() to decide emit-vs-reschedule, so a fake clock with a real Date
    // reschedules forever. rAF stays real so zoneless CD scheduling is not
    // frozen; the explicit useRealTimers() is mandatory because the global
    // vi.restoreAllMocks() does not undo fake timers.
    beforeEach(() => {
        vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval', 'Date'] });
    });
    afterEach(() => {
        vi.useRealTimers();
    });

    it('creates ONE "Custom Settings" derived from the readonly default on first edit', () => {
        const sysDefault = tpl({
            id: 'mask_default_sam3', name: 'Default', is_default: true, readonly: true,
            config: { text_prompt: 'subject', multimask_output: true },
        });
        const c = build([sysDefault]);
        expect(c.activeTemplateId()).toBe('mask_default_sam3');

        c.updateParam('text_prompt', 'person');
        vi.runAllTimers();

        expect(svc.createMaskingTemplate).toHaveBeenCalledTimes(1);
        const created = svc.createMaskingTemplate.mock.calls[0][0];
        expect(created.name).toBe('Custom Settings');
        expect(created.model_id).toBe('sam3');
        // Derived from the SYSTEM DEFAULT's config + the edit.
        expect(created.config).toMatchObject({ text_prompt: 'person', multimask_output: true });
        expect(c.activeTemplateId()).toBe('new-id');
        // The default itself was never written.
        expect(svc.updateTemplate).not.toHaveBeenCalledWith('masking', 'mask_default_sam3', expect.anything());
    });

    it('reuses an existing "Custom Settings" instead of creating a duplicate', () => {
        const sysDefault = tpl({ id: 'def', name: 'Default', is_default: true, readonly: true, config: { text_prompt: 'subject' } });
        const custom = tpl({ id: 'cs1', name: 'Custom Settings', project_id: 'p1', config: { text_prompt: 'hair' } });
        const c = build([sysDefault, custom], { active_mask_template: 'def' });
        expect(c.activeTemplateId()).toBe('def');

        c.updateParam('text_prompt', 'person');
        vi.runAllTimers();

        expect(svc.createMaskingTemplate).not.toHaveBeenCalled();
        expect(svc.updateTemplate).toHaveBeenCalledTimes(1);
        expect(svc.updateTemplate.mock.calls[0][1]).toBe('cs1');
        expect(c.activeTemplateId()).toBe('cs1');
    });

    it('never edits an is_default template in place even when not readonly', () => {
        const sysDefault = tpl({ id: 'def', name: 'Default', is_default: true, readonly: false, config: {} });
        const c = build([sysDefault]);

        c.updateParam('text_prompt', 'person');
        vi.runAllTimers();

        expect(svc.updateTemplate).not.toHaveBeenCalledWith('masking', 'def', expect.anything());
        expect(svc.createMaskingTemplate).toHaveBeenCalledTimes(1);
    });

    it('creates "Custom Settings" from current params when the model has NO template (RemBG gap)', () => {
        const c = build([]); // no templates seeded for this model
        expect(c.activeTemplateId()).toBeNull();

        c.updateParam('text_prompt', 'person');
        vi.runAllTimers();

        // The edit must not be silently dropped.
        expect(svc.createMaskingTemplate).toHaveBeenCalledTimes(1);
        expect(svc.createMaskingTemplate.mock.calls[0][0].config).toMatchObject({ text_prompt: 'person' });
        expect(c.activeTemplateId()).toBe('new-id');
    });

    it('does not create a second copy while the first create is in flight (slider drag)', () => {
        const sysDefault = tpl({ id: 'def', name: 'Default', is_default: true, readonly: true, config: {} });
        const c = build([sysDefault]);
        const create$ = new Subject<Template>();
        svc.createMaskingTemplate.mockReturnValue(create$.asObservable());

        c.updateParam('max_hole_area', 10);
        c.updateParam('max_hole_area', 20); // next slider tick, POST still pending

        expect(svc.createMaskingTemplate).toHaveBeenCalledTimes(1);

        create$.next(tpl({ id: 'new-id', name: 'Custom Settings' }));
        create$.complete();
        vi.runAllTimers();

        // The buffered edit lands in the created copy.
        expect(svc.updateTemplate).toHaveBeenCalledTimes(1);
        expect(svc.updateTemplate.mock.calls[0][1]).toBe('new-id');
        expect(svc.updateTemplate.mock.calls[0][2].config).toMatchObject({ max_hole_area: 20 });
    });

    it('still updates an editable user template directly', () => {
        const mine = tpl({ id: 'mine', name: 'My Masks', config: { text_prompt: 'hair' } });
        const c = build([mine], { active_mask_template: 'mine' });

        c.updateParam('text_prompt', 'face');
        vi.runAllTimers();

        expect(svc.createMaskingTemplate).not.toHaveBeenCalled();
        expect(svc.updateTemplate).toHaveBeenCalledWith('masking', 'mine',
            expect.objectContaining({ config: expect.objectContaining({ text_prompt: 'face' }) }));
    });

    it('falls back to the model code defaults when no template exists (no stale params)', () => {
        const c = build([]);
        // sam3 code defaults — not leftovers from a previous model.
        expect(c.maskingParams()).toMatchObject({ text_prompt: 'subject', multimask_output: true });
    });
});
