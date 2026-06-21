/**
 * IdeogramCaptionEditorComponent
 *
 * Sectioned editor for Ideogram 4 structured captions.
 * The compact JSON string (value model) is the single source of truth —
 * every field edit mutates → serializes → writes back to value().
 * doc() is a computed() derived from value() — always consistent, no effect timing issues.
 *
 * Binding strategy: use [value] property binding (not [ngModel]) for inputs/textareas
 * so DOM values are updated synchronously during change detection without NgModel overhead.
 * Events are captured via (input)/(change) to avoid FormsModule in zoneless test environments.
 * FormsModule is kept in imports only for the raw-JSON textarea's [(ngModel)] (two-way sync).
 */
import {
    ChangeDetectionStrategy,
    Component,
    computed,
    input,
    model,
    signal,
} from '@angular/core';
import { FormsModule } from '@angular/forms';
import {
    CANONICAL_MEDIUMS,
    MAX_ELEMENT_PALETTE,
    MAX_IMAGE_PALETTE,
    PHOTO_MEDIUM,
    type IdeogramCaption,
    type IdeogramElement,
    type IdeogramStyle,
    normalize,
    normalizeColor,
    parse,
    serialize,
} from './ideogram-format';
import { BboxOverlayComponent, type BboxItem } from './bbox-overlay';

// ---------------------------------------------------------------------------
// Internal working model
// ---------------------------------------------------------------------------

export interface WorkingDoc {
    description: string;
    style: IdeogramStyle;
    background: string;
    elements: IdeogramElement[];
}

function toWorking(caption: IdeogramCaption): WorkingDoc {
    return {
        description: caption.high_level_description,
        style: { ...caption.style_description, color_palette: [...caption.style_description.color_palette] },
        background: caption.compositional_deconstruction.background,
        elements: caption.compositional_deconstruction.elements.map(el => ({
            ...el,
            color_palette: [...el.color_palette],
            bbox: el.bbox ? [...el.bbox] : undefined,
        })),
    };
}

function fromWorking(doc: WorkingDoc): IdeogramCaption {
    return normalize({
        high_level_description: doc.description,
        style_description: { ...doc.style },
        compositional_deconstruction: {
            background: doc.background,
            elements: doc.elements,
        },
    });
}

const DEFAULT_DOC: WorkingDoc = toWorking(normalize({}));

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

@Component({
    selector: 'app-ideogram-caption-editor',
    standalone: true,
    changeDetection: ChangeDetectionStrategy.OnPush,
    imports: [FormsModule, BboxOverlayComponent],
    styles: [`
        :host { display: block; }
        .section-header {
            font-size: 10px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            cursor: pointer;
        }
        .field-label { font-size: 10px; color: var(--color-text-subtle, #888); margin-bottom: 2px; }
        .field-input {
            width: 100%;
            background: transparent;
            border: 1px solid var(--color-surface-high, #333);
            border-radius: 4px;
            color: var(--color-text-secondary, #ccc);
            font-size: 11px;
            padding: 4px 6px;
            resize: none;
        }
        .field-input:focus { outline: none; border-color: var(--color-brand, #6366f1); }
        .swatch {
            width: 18px; height: 18px; border-radius: 3px; border: 1px solid rgba(255,255,255,0.15);
            cursor: pointer; flex-shrink: 0;
        }
        .element-card {
            border: 1px solid var(--color-surface-high, #333);
            border-radius: 6px;
            padding: 8px;
        }
    `],
    template: `
        <!-- ===== High-level Description ===== -->
        <div class="px-3 py-2 border-b border-surface-mid">
            <div class="field-label">High-level description</div>
            <textarea
                data-testid="hld-textarea"
                class="field-input"
                rows="3"
                [value]="doc().description"
                (input)="onDescChange($any($event.target).value)"
            ></textarea>
        </div>

        <!-- ===== Style ===== -->
        <details open class="border-b border-surface-mid">
            <summary class="section-header px-3 py-2 text-text-subtle hover:text-brand transition-colors list-none flex items-center justify-between">
                <span>Style</span>
                <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/></svg>
            </summary>
            <div class="px-3 pb-3 space-y-2">
                <!-- Aesthetics -->
                <div>
                    <div class="field-label">Aesthetics</div>
                    <input type="text" class="field-input"
                        data-testid="style-aesthetics"
                        [value]="doc().style.aesthetics"
                        (input)="onStyleChange('aesthetics', $any($event.target).value)" />
                </div>
                <!-- Lighting -->
                <div>
                    <div class="field-label">Lighting</div>
                    <input type="text" class="field-input"
                        data-testid="style-lighting"
                        [value]="doc().style.lighting"
                        (input)="onStyleChange('lighting', $any($event.target).value)" />
                </div>
                <!-- Medium -->
                <div>
                    <div class="field-label">Medium</div>
                    <select class="field-input"
                        data-testid="style-medium"
                        [value]="mediumSelectValue()"
                        (change)="onMediumChange($any($event.target).value)">
                        @for (m of canonicalMediums; track m) {
                            <option [value]="m">{{ m }}</option>
                        }
                        <option value="__custom__">Custom…</option>
                    </select>
                </div>
                <!-- Custom medium input — shown when Custom is selected -->
                @if (mediumSelectValue() === '__custom__') {
                    <div>
                        <div class="field-label">Custom medium value</div>
                        <input type="text" class="field-input"
                            data-testid="style-medium-custom"
                            [value]="doc().style.medium"
                            (input)="onMediumChange($any($event.target).value)" />
                    </div>
                }
                <!-- Render field (photo / art_style) -->
                <div>
                    <div class="field-label" data-testid="render-field-label">
                        {{ isPhoto() ? 'Photo (camera / film)' : 'Art style (rendering technique)' }}
                    </div>
                    <input type="text" class="field-input"
                        data-testid="style-render"
                        [value]="renderFieldValue()"
                        (input)="onRenderChange($any($event.target).value)" />
                </div>
                <!-- Color palette -->
                <div>
                    <div class="field-label">Color palette (max {{ maxImagePalette }})</div>
                    <div class="flex flex-wrap gap-1 mb-1">
                        @for (c of doc().style.color_palette; track c; let i = $index) {
                            <div class="flex items-center gap-0.5">
                                <div class="swatch" [style.background]="c" [title]="c"></div>
                                <button type="button" class="text-text-subtle hover:text-danger text-[10px] px-0.5"
                                    [attr.data-testid]="'palette-remove-' + i"
                                    (click)="removePaletteColor(i)">×</button>
                            </div>
                        }
                    </div>
                    <div class="flex gap-1 items-center">
                        <input type="color" class="w-7 h-7 rounded cursor-pointer border-0 bg-transparent"
                            data-testid="palette-color-input"
                            [value]="newColor()"
                            (input)="newColor.set($any($event.target).value)" />
                        <input type="text" class="field-input flex-1"
                            data-testid="palette-hex-input"
                            placeholder="#RRGGBB"
                            [value]="newColor()"
                            (input)="newColor.set($any($event.target).value)" />
                        <button type="button"
                            data-testid="palette-add"
                            class="px-2 py-1 bg-surface-mid hover:bg-surface-high text-text-secondary text-[11px] rounded border border-surface-high/40"
                            (click)="addPaletteColor()">Add</button>
                    </div>
                </div>
            </div>
        </details>

        <!-- ===== Background ===== -->
        <div class="px-3 py-2 border-b border-surface-mid">
            <div class="field-label">Background</div>
            <textarea class="field-input" rows="2"
                data-testid="background-textarea"
                [value]="doc().background"
                (input)="onBackgroundChange($any($event.target).value)"></textarea>
        </div>

        <!-- ===== Elements ===== -->
        <details open class="border-b border-surface-mid">
            <summary class="section-header px-3 py-2 text-text-subtle hover:text-brand transition-colors list-none flex items-center justify-between">
                <span>Elements ({{ doc().elements.length }})</span>
                <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/></svg>
            </summary>
            <div class="px-3 pb-3 space-y-2">
                <!-- Bbox overlay -->
                <app-bbox-overlay
                    [imageUrl]="imageUrl() ?? ''"
                    [boxes]="bboxItems()"
                    [selectedId]="selectedElementId()"
                    [drawEnabled]="drawEnabled()"
                    (boxSelected)="selectedElementId.set($event)"
                    (boxAdded)="onBoxAdded($event)"
                />
                <div class="flex gap-1 mb-1">
                    <button type="button"
                        data-testid="draw-toggle"
                        class="px-2 py-1 text-[11px] rounded border transition-colors"
                        [class.bg-brand]="drawEnabled()"
                        [class.text-white]="drawEnabled()"
                        [class.bg-surface-mid]="!drawEnabled()"
                        [class.text-text-secondary]="!drawEnabled()"
                        [class.border-brand]="drawEnabled()"
                        [class.border-surface-high]="!drawEnabled()"
                        (click)="drawEnabled.set(!drawEnabled())">
                        {{ drawEnabled() ? 'Done Drawing' : 'Draw box' }}
                    </button>
                    <button type="button"
                        data-testid="add-element"
                        class="px-2 py-1 bg-surface-mid hover:bg-surface-high text-text-secondary text-[11px] rounded border border-surface-high/40"
                        (click)="addElement()">+ Add Element</button>
                </div>

                @for (el of doc().elements; track $index; let i = $index) {
                    <div class="element-card" data-testid="element-card" [attr.data-element-index]="i">
                        <!-- Type toggle -->
                        <div class="flex items-center gap-2 mb-2">
                            <button type="button" class="text-[10px] px-2 py-0.5 rounded border transition-colors"
                                [class.bg-brand]="el.type === 'obj'"
                                [class.text-white]="el.type === 'obj'"
                                [class.border-brand]="el.type === 'obj'"
                                [class.bg-surface-mid]="el.type !== 'obj'"
                                [class.text-text-subtle]="el.type !== 'obj'"
                                [class.border-surface-high]="el.type !== 'obj'"
                                (click)="setElementType(i, 'obj')">obj</button>
                            <button type="button" class="text-[10px] px-2 py-0.5 rounded border transition-colors"
                                [class.bg-brand]="el.type === 'text'"
                                [class.text-white]="el.type === 'text'"
                                [class.border-brand]="el.type === 'text'"
                                [class.bg-surface-mid]="el.type !== 'text'"
                                [class.text-text-subtle]="el.type !== 'text'"
                                [class.border-surface-high]="el.type !== 'text'"
                                (click)="setElementType(i, 'text')">text</button>
                            <button type="button" class="ml-auto text-[10px] text-text-subtle hover:text-danger"
                                [attr.data-testid]="'remove-element-' + i"
                                (click)="removeElement(i)">Remove</button>
                        </div>

                        <!-- Text field (only for text type) -->
                        @if (el.type === 'text') {
                            <div class="mb-2">
                                <div class="field-label">Text</div>
                                <input type="text" class="field-input"
                                    [attr.data-testid]="'element-text-' + i"
                                    [value]="el.text ?? ''"
                                    (input)="onElementTextChange(i, $any($event.target).value)" />
                            </div>
                        }

                        <!-- Description -->
                        <div class="mb-2">
                            <div class="field-label">Description</div>
                            <textarea class="field-input" rows="2"
                                [attr.data-testid]="'element-desc-' + i"
                                [value]="el.desc"
                                (input)="onElementDescChange(i, $any($event.target).value)"></textarea>
                        </div>

                        <!-- BBox (y1 x1 y2 x2) -->
                        <div class="mb-2">
                            <div class="field-label">BBox [y1, x1, y2, x2] (0–1000)</div>
                            <div class="flex gap-1">
                                @for (coord of ['y1','x1','y2','x2']; track coord; let ci = $index) {
                                    <input type="number" class="field-input text-center" min="0" max="1000"
                                        [attr.data-testid]="'element-bbox-' + i + '-' + coord"
                                        [value]="el.bbox ? el.bbox[ci] : 0"
                                        (input)="onElementBboxChange(i, ci, +$any($event.target).value)" />
                                }
                            </div>
                        </div>

                        <!-- Per-element color palette (max 5) -->
                        <div>
                            <div class="field-label">Colors (max {{ maxElementPalette }})</div>
                            <div class="flex flex-wrap gap-1 mb-1">
                                @for (c of el.color_palette; track c; let ci = $index) {
                                    <div class="flex items-center gap-0.5">
                                        <div class="swatch" [style.background]="c" [title]="c"></div>
                                        <button type="button" class="text-text-subtle hover:text-danger text-[10px] px-0.5"
                                            [attr.data-testid]="'element-color-remove-' + i + '-' + ci"
                                            (click)="removeElementColor(i, ci)">×</button>
                                    </div>
                                }
                            </div>
                            <div class="flex gap-1 items-center">
                                <input type="color" class="w-7 h-7 rounded cursor-pointer border-0 bg-transparent"
                                    [attr.data-testid]="'element-color-input-' + i"
                                    [value]="elementNewColors()[i] ?? '#000000'"
                                    (input)="onElementNewColorChange(i, $any($event.target).value)" />
                                <input type="text" class="field-input flex-1"
                                    [attr.data-testid]="'element-color-hex-' + i"
                                    placeholder="#RRGGBB"
                                    [value]="elementNewColors()[i] ?? '#000000'"
                                    (input)="onElementNewColorChange(i, $any($event.target).value)" />
                                <button type="button"
                                    [attr.data-testid]="'element-color-add-' + i"
                                    class="px-2 py-1 bg-surface-mid hover:bg-surface-high text-text-secondary text-[11px] rounded border border-surface-high/40"
                                    [disabled]="el.color_palette.length >= maxElementPalette"
                                    (click)="addElementColor(i, elementNewColors()[i] ?? '#000000')">Add</button>
                            </div>
                        </div>
                    </div>
                }
            </div>
        </details>

        <!-- ===== Raw JSON ===== -->
        <details class="border-b border-surface-mid">
            <summary class="section-header px-3 py-2 text-text-subtle hover:text-brand transition-colors list-none">
                Raw JSON
            </summary>
            <div class="px-3 pb-3">
                <textarea class="field-input font-mono text-[10px]" rows="8"
                    data-testid="raw-json-textarea"
                    [value]="value() ?? ''"
                    (input)="onRawJsonInput($any($event.target).value)"></textarea>
            </div>
        </details>
    `,
})
export class IdeogramCaptionEditorComponent {
    // Public API — two-way bound compact JSON string (single source of truth)
    readonly value = model<string>();
    readonly imageUrl = input<string>();

    // doc() is a computed() that always reflects value() — no effect timing issues.
    // User edits call commit() which writes value(), and doc() auto-updates via computed().
    protected readonly doc = computed<WorkingDoc>(() => {
        const v = this.value();
        if (!v) return DEFAULT_DOC;
        const parsed = parse(v);
        if (!parsed) return DEFAULT_DOC;
        return toWorking(normalize(parsed));
    });

    // UI state
    protected readonly newColor = signal<string>('#000000');
    /** Per-element pending color values (index → hex string). */
    protected readonly elementNewColors = signal<Partial<Record<number, string>>>({});
    protected readonly selectedElementId = signal<string | null>(null);
    protected readonly drawEnabled = signal<boolean>(false);

    protected readonly canonicalMediums = CANONICAL_MEDIUMS;
    protected readonly maxImagePalette = MAX_IMAGE_PALETTE;
    protected readonly maxElementPalette = MAX_ELEMENT_PALETTE;

    /** True when the current medium is 'photograph' */
    protected readonly isPhoto = computed(() => this.doc().style.medium === PHOTO_MEDIUM);

    /** Which value to show in the <select> — '__custom__' for non-canonical mediums */
    protected readonly mediumSelectValue = computed(() => {
        const m = this.doc().style.medium;
        return (CANONICAL_MEDIUMS as readonly string[]).includes(m) ? m : '__custom__';
    });

    /** The value of the render field (photo or art_style, whichever is set) */
    protected readonly renderFieldValue = computed(() => {
        const s = this.doc().style;
        return s.photo != null ? s.photo : (s.art_style ?? '');
    });

    /** Bbox items derived from elements for the overlay */
    protected readonly bboxItems = computed<BboxItem[]>(() =>
        this.doc().elements
            .map((el, i) => el.bbox ? { id: String(i), bbox: el.bbox } : null)
            .filter((x): x is BboxItem => x !== null)
    );

    // ---------------------------------------------------------------------------
    // Commit — write serialized value back; doc() auto-updates via computed()
    // ---------------------------------------------------------------------------

    private commit(updatedDoc: WorkingDoc): void {
        this.value.set(serialize(fromWorking(updatedDoc)));
    }

    // ---------------------------------------------------------------------------
    // Field mutation handlers
    // ---------------------------------------------------------------------------

    protected onDescChange(val: string): void {
        this.commit({ ...this.doc(), description: val });
    }

    protected onStyleChange(field: keyof IdeogramStyle, val: string): void {
        this.commit({ ...this.doc(), style: { ...this.doc().style, [field]: val } });
    }

    protected onMediumChange(val: string): void {
        if (val === '__custom__') return; // wait for free-form input
        const oldStyle = this.doc().style;
        // Gather existing render value before re-normalizing
        const existingRender = oldStyle.photo != null ? oldStyle.photo : (oldStyle.art_style ?? '');
        // Build a raw style object with the new medium and the existing render value;
        // normalize() will route render into the correct branch key (photo / art_style)
        const rawStyle: Record<string, unknown> = {
            aesthetics: oldStyle.aesthetics,
            lighting: oldStyle.lighting,
            medium: val,
            color_palette: oldStyle.color_palette,
        };
        if (val === PHOTO_MEDIUM) {
            rawStyle['photo'] = existingRender;
        } else {
            rawStyle['art_style'] = existingRender;
        }
        const normalized = normalize({
            high_level_description: this.doc().description,
            style_description: rawStyle,
            compositional_deconstruction: {
                background: this.doc().background,
                elements: this.doc().elements,
            },
        });
        this.value.set(serialize(normalized));
    }

    protected onRenderChange(val: string): void {
        const style = this.doc().style;
        const updatedStyle = this.isPhoto()
            ? { ...style, photo: val, art_style: undefined }
            : { ...style, art_style: val, photo: undefined };
        this.commit({ ...this.doc(), style: updatedStyle });
    }

    protected onBackgroundChange(val: string): void {
        this.commit({ ...this.doc(), background: val });
    }

    // Image-level palette
    protected addPaletteColor(): void {
        const nc = normalizeColor(this.newColor());
        if (!nc) return;
        const palette = this.doc().style.color_palette;
        if (palette.length >= MAX_IMAGE_PALETTE) return;
        if (palette.includes(nc)) return;
        this.commit({ ...this.doc(), style: { ...this.doc().style, color_palette: [...palette, nc] } });
    }

    protected removePaletteColor(index: number): void {
        const palette = this.doc().style.color_palette.filter((_, i) => i !== index);
        this.commit({ ...this.doc(), style: { ...this.doc().style, color_palette: palette } });
    }

    // Elements
    protected addElement(): void {
        const newEl: IdeogramElement = { type: 'obj', desc: '', color_palette: [] };
        this.commit({ ...this.doc(), elements: [...this.doc().elements, newEl] });
    }

    protected removeElement(index: number): void {
        const elements = this.doc().elements.filter((_, i) => i !== index);
        this.commit({ ...this.doc(), elements });
    }

    protected setElementType(index: number, type: 'obj' | 'text'): void {
        const elements = this.doc().elements.map((el, i) =>
            i === index ? { ...el, type } : el
        );
        this.commit({ ...this.doc(), elements });
    }

    protected onElementTextChange(index: number, val: string): void {
        const elements = this.doc().elements.map((el, i) =>
            i === index ? { ...el, text: val } : el
        );
        this.commit({ ...this.doc(), elements });
    }

    protected onElementDescChange(index: number, val: string): void {
        const elements = this.doc().elements.map((el, i) =>
            i === index ? { ...el, desc: val } : el
        );
        this.commit({ ...this.doc(), elements });
    }

    protected onElementBboxChange(elIdx: number, coordIdx: number, val: number): void {
        const elements = this.doc().elements.map((el, i) => {
            if (i !== elIdx) return el;
            const bbox = el.bbox ? [...el.bbox] : [0, 0, 0, 0];
            bbox[coordIdx] = Math.max(0, Math.min(1000, Math.round(Number(val) || 0)));
            return { ...el, bbox };
        });
        this.commit({ ...this.doc(), elements });
    }

    protected onElementNewColorChange(elIdx: number, val: string): void {
        this.elementNewColors.update(m => ({ ...m, [elIdx]: val }));
    }

    protected addElementColor(elementIndex: number, color: string): void {
        const nc = normalizeColor(color);
        if (!nc) return;
        const palette = this.doc().elements[elementIndex]?.color_palette ?? [];
        if (palette.length >= MAX_ELEMENT_PALETTE) return;
        if (palette.includes(nc)) return;
        const elements = this.doc().elements.map((el, i) =>
            i === elementIndex ? { ...el, color_palette: [...el.color_palette, nc] } : el
        );
        this.commit({ ...this.doc(), elements });
    }

    protected removeElementColor(elIdx: number, colorIdx: number): void {
        const elements = this.doc().elements.map((el, i) => {
            if (i !== elIdx) return el;
            return { ...el, color_palette: el.color_palette.filter((_, ci) => ci !== colorIdx) };
        });
        this.commit({ ...this.doc(), elements });
    }

    protected onBoxAdded(bbox: number[]): void {
        const newEl: IdeogramElement = { type: 'obj', bbox, desc: '', color_palette: [] };
        const elements = [...this.doc().elements, newEl];
        this.commit({ ...this.doc(), elements });
        this.selectedElementId.set(String(elements.length - 1));
        this.drawEnabled.set(false);
    }

    // Raw JSON editing
    protected onRawJsonInput(rawText: string): void {
        const parsed = parse(rawText);
        if (parsed) {
            // Valid JSON — update value; doc() will recompute automatically
            this.value.set(serialize(normalize(parsed)));
        }
        // If invalid JSON — do nothing (value unchanged, no crash)
    }
}
