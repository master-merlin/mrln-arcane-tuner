import { TestBed } from '@angular/core/testing';
import { signal } from '@angular/core';
import { of } from 'rxjs';
import { ServerControlComponent } from './server-control';
import { ToastService } from '../../../services/toast';
import { ModelService } from '../../../services/model.service';
import { FilesystemService } from '../../../services/filesystem.service';
import { SettingsStore } from '../../../state/settings.store';
import { SystemService } from '../../../services/system.service';

/**
 * The Backend Port field explains itself in a container.
 *
 * WHY THIS EXISTS (DECISION-11 (a), LANE-13). `resolve_port` treats the
 * settings file as not a port source at all when `MRLN_CONTAINER=1`: the port
 * comes from argv or `PORT`, and the host side of `docker run -p` lives in the
 * daemon where nothing inside the container can read it. So an operator who
 * edits this field in a pod is editing a value nothing reads. Before this note
 * the screen gave them no way to know that.
 *
 * The ENFORCEMENT stays in the backend resolver. This component only explains
 * it — "consolidating" the rule into the screen is the tidy-up someone makes
 * in six months, and it would put the guarantee behind a UI that a scripted
 * client never loads.
 */
describe('ServerControlComponent — container port note', () => {
    let containerMode: ReturnType<typeof signal<boolean>>;

    function create() {
        const fixture = TestBed.createComponent(ServerControlComponent);
        fixture.detectChanges();
        // Second pass: with template-driven forms `[disabled]` reaches the
        // element through NgModel's setDisabledState, one change-detection
        // cycle after the binding itself.
        fixture.detectChanges();
        return fixture;
    }

    /** Text of the Backend Port field ONLY. The sibling Frontend Port field
     *  carries its own "requires restart" hint, so asserting against the whole
     *  component's textContent tests the wrong field and passes for the wrong
     *  reason. */
    function backendPortField(fixture: ReturnType<typeof create>): HTMLElement {
        const input = fixture.nativeElement.querySelector('[data-testid="setting-backend-port"]');
        return input.closest('.sc-field') as HTMLElement;
    }

    function hint(fixture: ReturnType<typeof create>, testid: string): HTMLElement | null {
        return fixture.nativeElement.querySelector(`[data-testid="${testid}"]`);
    }

    beforeEach(() => {
        containerMode = signal(false);
        TestBed.configureTestingModule({
            imports: [ServerControlComponent],
            providers: [
                { provide: ToastService, useValue: { success: vi.fn(), error: vi.fn() } },
                {
                    provide: ModelService,
                    useValue: { getModelSettings: vi.fn(() => of({})) },
                },
                { provide: FilesystemService, useValue: {} },
                {
                    provide: SettingsStore,
                    useValue: {
                        byId: () => () => ({
                            settings: {
                                backend_port: 8000,
                                frontend_port: 4200,
                                log_level: 'INFO',
                                start_frontend: false,
                            },
                        }),
                        loadModule: vi.fn(() => Promise.resolve()),
                        updateModule: vi.fn(() => Promise.resolve()),
                    },
                },
                {
                    provide: SystemService,
                    useValue: {
                        containerMode,
                        getVersion: vi.fn(() => of({ version: '0.0.0', container: false })),
                    },
                },
            ],
        });
    });

    it('a local install keeps the editable field and the restart hint', () => {
        // The positive control, and it is the one that matters: without it
        // every assertion below is satisfied by a component that renders the
        // container note unconditionally.
        const fixture = create();

        expect(hint(fixture, 'backend-port-container-note')).toBeNull();
        const input = hint(fixture, 'setting-backend-port') as HTMLInputElement;
        expect(input).not.toBeNull();
        expect(input.readOnly).toBe(false);
        expect(backendPortField(fixture).textContent).toContain('requires restart');
    });

    it('a container replaces the hint and locks the field', () => {
        containerMode.set(true);
        const fixture = create();

        const note = hint(fixture, 'backend-port-container-note');
        expect(note).not.toBeNull();
        expect(note!.textContent).toContain('set by the platform');

        const input = hint(fixture, 'setting-backend-port') as HTMLInputElement;
        expect(input.readOnly).toBe(true);
        // `readonly`, not `disabled`: NgModel's value accessor owns the
        // `disabled` property and resets it each change-detection pass, so
        // that binding is silently undone. Measured, not assumed - the
        // first two drafts of this line used `disabled` and failed here.
    });

    it('the two hints are mutually exclusive, not merely both present', () => {
        // Asserting only that the note appears would pass a template that
        // shows "requires restart" underneath it — two contradictory
        // instructions stacked, which is worse than either alone.
        containerMode.set(true);
        const fixture = create();

        expect(backendPortField(fixture).textContent).not.toContain('requires restart');
        expect(backendPortField(fixture).textContent).toContain('set by the platform');
    });

    it('refreshes the flag itself rather than trusting the shell', () => {
        // If the shell's startup call failed, the signal still holds its
        // `false` default and this screen would offer a control the container
        // ignores. Self-healing per the derived-data invariant.
        const fixture = create();
        const system = TestBed.inject(SystemService) as unknown as {
            getVersion: ReturnType<typeof vi.fn>;
        };
        expect(system.getVersion).toHaveBeenCalled();
        expect(fixture).toBeTruthy();
    });
});
