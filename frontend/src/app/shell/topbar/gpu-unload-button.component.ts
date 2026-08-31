import { ChangeDetectionStrategy, Component, computed, inject } from '@angular/core';
import { IcoComponent } from '../../icons/ico.component';
import { GpuResidencyStore } from '../../state/gpu-residency.store';
import { ToastService } from '../../services/toast';

/**
 * Topbar "free the GPU" control (ComfyUI's "Unload models", for this app).
 *
 * **Positive-only**, exactly like the LLM button beside it: hidden unless the
 * backend says something is actually resident. A control that is always there
 * would be a control that mostly does nothing, and the topbar is the one place
 * in the app where every pixel is spent on state the user should react to.
 *
 * The tooltip names what is loaded, because "unload models" alone does not tell
 * the user whether pressing it costs them the caption model they are mid-loop
 * with or a scoring model they are done with.
 */
@Component({
    selector: 'app-gpu-unload-button',
    standalone: true,
    imports: [IcoComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        @if (gpu.anyLoaded()) {
            <button class="icon-btn" type="button"
                    data-testid="gpu-unload-btn"
                    [title]="tooltip()"
                    [attr.aria-label]="tooltip()"
                    [attr.aria-busy]="gpu.unloading() ? 'true' : null"
                    [class.is-busy]="gpu.unloading()"
                    [disabled]="gpu.unloading()"
                    (click)="onUnload()">
                <app-ico name="MemoryStick" [size]="15" />
            </button>
        }
    `,
    styles: [`
        .icon-btn.is-busy { opacity: .55; cursor: progress; }
    `],
})
export class GpuUnloadButtonComponent {
    protected gpu = inject(GpuResidencyStore);
    private toast = inject(ToastService);

    protected loadedLabels = computed(() =>
        this.gpu.services().filter(s => s.loaded).map(s => s.label),
    );

    protected tooltip = computed(() => {
        if (this.gpu.unloading()) return 'Freeing GPU memory…';
        const names = this.loadedLabels();
        if (!names.length) return 'Free GPU memory';
        return `Free GPU memory — ${names.join(', ')} loaded`;
    });

    protected onUnload(): void {
        if (this.gpu.unloading()) return;
        this.gpu.unloadAll().subscribe({
            next: r => {
                const freed = r.unloaded.length;
                const skipped = r.skipped;
                if (freed && skipped.length) {
                    this.toast.warning(
                        `Freed ${freed} model(s). Kept: ${skipped.map(s => s.reason).join('; ')}.`,
                    );
                } else if (skipped.length) {
                    this.toast.warning(`Nothing freed — ${skipped.map(s => s.reason).join('; ')}.`);
                } else if (freed) {
                    this.toast.success(`Freed ${freed} model(s) — GPU memory released.`);
                } else {
                    this.toast.info('No models were loaded.');
                }
            },
            error: () => this.toast.error('Could not free GPU memory. Check the server logs.'),
        });
    }
}
