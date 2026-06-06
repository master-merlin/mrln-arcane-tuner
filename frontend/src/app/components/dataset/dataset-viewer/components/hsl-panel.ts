import { Component, input, output, signal, ChangeDetectionStrategy } from '@angular/core';

export interface HSLRangeAdjustment {
    hue_shift: number;
    saturation: number;
    luminance: number;
}

export interface HSLConfig {
    [range: string]: HSLRangeAdjustment;
}

interface HSLRange {
    key: string;
    label: string;
    color: string;
    activeClass: string;
}

@Component({
    selector: 'app-hsl-panel',
    standalone: true,
    changeDetection: ChangeDetectionStrategy.OnPush,
    imports: [],
    template: `
    <div class="flex flex-col gap-3">
        <!-- Range selector -->
        <div class="flex flex-wrap gap-1">
            @for (range of ranges; track range.key) {
                <button
                    (click)="selectRange(range.key)"
                    [class]="'w-6 h-6 rounded-full text-[8px] font-bold flex items-center justify-center transition-all border-2 ' +
                        (activeRange() === range.key
                            ? 'border-white/80 scale-110 shadow-lg'
                            : 'border-transparent hover:border-white/30')"
                    [style.background]="range.color"
                    [attr.data-testid]="'hsl-range-' + range.key">
                </button>
            }
        </div>

        <!-- Active range label -->
        <div class="flex items-center justify-between px-1">
            <span class="text-[11px] font-medium text-text capitalize">{{ activeRange() }}</span>
            <button (click)="resetRange()" class="text-[10px] text-text-muted hover:text-red-400 transition-colors" data-testid="hsl-reset-range">
                Reset
            </button>
        </div>

        <!-- Hue Shift -->
        <div class="flex flex-col gap-1">
            <div class="flex items-center justify-between">
                <label class="text-[10px] text-text-muted uppercase tracking-wider">Hue</label>
                <span class="text-[10px] font-mono text-text-muted">{{ getAdj('hue_shift') }}°</span>
            </div>
            <input type="range" min="-30" max="30" step="1"
                [value]="getAdj('hue_shift')"
                (input)="onSlider('hue_shift', $event)"
                class="w-full accent-brand cursor-pointer" data-testid="hsl-hue-slider">
        </div>

        <!-- Saturation -->
        <div class="flex flex-col gap-1">
            <div class="flex items-center justify-between">
                <label class="text-[10px] text-text-muted uppercase tracking-wider">Saturation</label>
                <span class="text-[10px] font-mono text-text-muted">{{ getAdj('saturation') }}</span>
            </div>
            <input type="range" min="-100" max="100" step="1"
                [value]="getAdj('saturation')"
                (input)="onSlider('saturation', $event)"
                class="w-full accent-brand cursor-pointer" data-testid="hsl-sat-slider">
        </div>

        <!-- Luminance -->
        <div class="flex flex-col gap-1">
            <div class="flex items-center justify-between">
                <label class="text-[10px] text-text-muted uppercase tracking-wider">Luminance</label>
                <span class="text-[10px] font-mono text-text-muted">{{ getAdj('luminance') }}</span>
            </div>
            <input type="range" min="-100" max="100" step="1"
                [value]="getAdj('luminance')"
                (input)="onSlider('luminance', $event)"
                class="w-full accent-brand cursor-pointer" data-testid="hsl-lum-slider">
        </div>
    </div>
    `,
    styles: []
})
export class HSLPanelComponent {
    hslConfig = input<HSLConfig>({});
    hslChanged = output<HSLConfig>();

    activeRange = signal<string>('reds');

    ranges: HSLRange[] = [
        { key: 'reds', label: 'R', color: '#e53e3e', activeClass: '' },
        { key: 'oranges', label: 'O', color: '#dd6b20', activeClass: '' },
        { key: 'yellows', label: 'Y', color: '#d69e2e', activeClass: '' },
        { key: 'greens', label: 'G', color: '#38a169', activeClass: '' },
        { key: 'cyans', label: 'C', color: '#0bc5ea', activeClass: '' },
        { key: 'blues', label: 'B', color: '#3182ce', activeClass: '' },
        { key: 'purples', label: 'P', color: '#805ad5', activeClass: '' },
        { key: 'magentas', label: 'M', color: '#d53f8c', activeClass: '' },
    ];

    selectRange(key: string): void {
        this.activeRange.set(key);
    }

    getAdj(field: keyof HSLRangeAdjustment): number {
        const config = this.hslConfig();
        const range = config[this.activeRange()];
        if (!range) return 0;
        return range[field] ?? 0;
    }

    onSlider(field: string, event: Event): void {
        const value = +(event.target as HTMLInputElement).value;
        const config = { ...this.hslConfig() };
        const current = config[this.activeRange()] || { hue_shift: 0, saturation: 0, luminance: 0 };
        config[this.activeRange()] = { ...current, [field]: value };
        this.hslChanged.emit(config);
    }

    resetRange(): void {
        const config = { ...this.hslConfig() };
        config[this.activeRange()] = { hue_shift: 0, saturation: 0, luminance: 0 };
        this.hslChanged.emit(config);
    }
}
