import { ChangeDetectionStrategy, Component, signal } from '@angular/core';
import { LoraToolsComponent, type ToolTab } from '../../components/tools/lora-tools/lora-tools';
import { TabsComponent, type TabItem } from '../../ui/tabs/tabs.component';

/**
 * Tools screen — header + Inspect/Resize tabs over the `lora-tools` component.
 * The outer tabs drive `lora-tools` via its `tab` input (single mount, so
 * inspect results survive a toggle to Resize and back). Merge / Speed Train
 * are not surfaced until they're actually built.
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
    ];
}
