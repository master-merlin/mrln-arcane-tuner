import { ChangeDetectionStrategy, Component, signal } from '@angular/core';
import { LoraToolsComponent } from '../../components/tools/lora-tools/lora-tools';
import { TabsComponent, type TabItem } from '../../ui/tabs/tabs.component';

type ToolTab = 'inspect' | 'resize' | 'merge' | 'speed';

/**
 * Tools screen — wraps the existing `lora-tools` component for Inspect +
 * Resize. Merge and Speed Train are deferred (visible as disabled "coming
 * soon" tabs per spec §4.6).
 *
 * The wrapped `<app-lora-tools/>` carries its own internal sub-tab signal
 * for inspect vs. resize, so clicking the outer Inspect/Resize tabs is
 * purely cosmetic — the component renders both flows from a single mount.
 * Switching them here without modifying the legacy component is not
 * possible; the disabled outer chips make the future shape explicit.
 *
 * TODO(frontend): LoRA cards inside lora-tools should display a version
 * chip equal to the producing dataset's version (spec §4.6). The current
 * lora-tools template doesn't render LoRA cards (it operates on a single
 * file path at a time), so there's no card to extend yet.
 */
@Component({
    selector: 'app-tools-screen',
    standalone: true,
    imports: [LoraToolsComponent, TabsComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    templateUrl: './tools-screen.html',
})
export class ToolsScreen {
    protected tab = signal<ToolTab>('inspect');

    protected tabs: TabItem<ToolTab>[] = [
        { value: 'inspect', label: 'Inspect' },
        { value: 'resize', label: 'Resize' },
        { value: 'merge', label: 'Merge (coming soon)', disabled: true },
        { value: 'speed', label: 'Speed Train (coming soon)', disabled: true },
    ];
}
