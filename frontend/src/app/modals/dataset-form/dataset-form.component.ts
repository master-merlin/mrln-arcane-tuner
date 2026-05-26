import {
    ChangeDetectionStrategy, Component, computed, effect, inject, signal,
} from '@angular/core';
import { toSignal } from '@angular/core/rxjs-interop';
import { AbstractControl, FormBuilder, ReactiveFormsModule } from '@angular/forms';
import { IcoComponent } from '../../icons/ico.component';
import { DatasetStore } from '../../state/dataset.store';
import { OverlayStore } from '../../state/overlay.store';

const STANDARD_CLASSIFIERS = ['vehicle', 'person', 'style', 'object', 'landscape'] as const;
const NAME_FORBIDDEN = /[<>:"/\\|?*]/;
/** Sentinel select-option value that switches the category control into inline-text-entry mode. */
const CUSTOM_CLASSIFIER_KEY = '__custom__';

const LEET_LIGHT: Record<string, string> = {
    a: '4', A: '4', e: '3', E: '3', i: '1', I: '1', o: '0', O: '0',
};
const LEET_FULL: Record<string, string> = {
    ...LEET_LIGHT,
    s: '5', S: '5', t: '7', T: '7',
};

function stripSeparators(s: string): string {
    return s.replace(/[^A-Za-z0-9]+/g, '');
}

/**
 * Trigger-word generation strategies. Each takes the raw (untrimmed)
 * name and returns a candidate. Empty results mean "not applicable for
 * this input" and the caller should skip to the next strategy.
 *
 * The wand button cycles through these on repeat clicks, giving the
 * user alternative phrasings without forcing them to invent one.
 */
const TRIGGER_STRATEGIES: ReadonlyArray<(raw: string) => string> = [
    // 0 — Leet (light): strip seps, first vowel → leet number
    //     "911 Targa" → "911T4rga"
    (raw) => {
        const s = stripSeparators(raw);
        const m = /[aeioAEIO]/.exec(s);
        return m ? s.slice(0, m.index) + LEET_LIGHT[m[0]] + s.slice(m.index + 1) : s;
    },
    // 1 — Leet (full): strip seps, replace all leet-able letters (a/e/i/o/s/t)
    //     "911 Targa" → "9174rg4", "My Style" → "My57yl3"
    (raw) => stripSeparators(raw).replace(/[aeiostAEIOST]/g, (ch) => LEET_FULL[ch] ?? ch),
    // 2 — Compact: strip seps, preserve case
    //     "911 Targa" → "911Targa"
    (raw) => stripSeparators(raw),
    // 3 — Lowercase compact: strip seps, lowercase
    //     "911 Targa" → "911targa"
    (raw) => stripSeparators(raw).toLowerCase(),
    // 4 — Initials: first letter of each alpha token + whole digit tokens
    //     "Porsche 911 Targa" → "P911T", "Mercedes Benz 300SL" → "MB300SL"
    //     Skipped (empty return) when the result would collapse to <2 chars.
    (raw) => {
        const tokens = raw.split(/[^A-Za-z0-9]+/).filter(Boolean);
        if (tokens.length === 0) return '';
        const out = tokens.map((t) => (/^\d/.test(t) ? t : t.charAt(0).toUpperCase())).join('');
        return out.length >= 2 ? out : '';
    },
];

/** Rejects null/empty/whitespace-only values. `Validators.required` accepts "   " — this does not. */
function nonEmptyTrimmed(c: AbstractControl): { required: true } | null {
    const v = c.value as string | null;
    return (v ?? '').trim().length === 0 ? { required: true } : null;
}

interface DatasetFormModalData {
    /** When set, modal runs in edit mode against this dataset id (or name as fallback). */
    datasetId?: string;
}

/**
 * Dataset form modal — context-aware Create / Edit dialog.
 *
 * The modal opens via `overlay.openModal('dataset-form')` for creation,
 * or `overlay.openModal('dataset-form', { datasetId })` for edit. In
 * edit mode the form is prefilled once from {@link DatasetStore} and
 * submits patch through `datasetStore.updateDataset`.
 *
 * Fields mirror the backend `UpdateDatasetRequest` payload — name +
 * classifier + description + trigger_word + tags + notes. The legacy
 * `dataset-form-modal` covered only the first three; the LoRA-flavored
 * three were added in the overhaul once migration v7 landed.
 */
@Component({
    selector: 'app-modal-dataset-form',
    standalone: true,
    imports: [ReactiveFormsModule, IcoComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="modal-head">
            <div>
                <div class="eyebrow">{{ isEdit() ? 'EDIT' : 'CREATE' }}</div>
                <div class="modal-title">{{ isEdit() ? 'Edit Dataset' : 'New Dataset' }}</div>
            </div>
            <button class="icon-btn" type="button" (click)="overlay.closeModal()" aria-label="Close">×</button>
        </div>

        <div class="modal-body">
            <form [formGroup]="form" (ngSubmit)="submit()">
                <label class="field-label">Name</label>
                <input class="input" type="text" formControlName="name" autocomplete="off"
                       placeholder="My_New_Concept" autofocus/>
                @if (nameInvalid()) {
                    <div class="field-error">Name contains forbidden characters (&lt; &gt; : " / \\ | ? *)</div>
                }

                <label class="field-label df-mt">Category</label>
                <select class="select" formControlName="classifier">
                    <option value="">None / Uncategorized</option>
                    <optgroup label="Standard">
                        @for (s of standardClassifiers; track s) {
                            <option [value]="s">{{ titlecase(s) }}</option>
                        }
                    </optgroup>
                    @if (reusableClassifiers().length > 0) {
                        <optgroup label="Previously used">
                            @for (c of reusableClassifiers(); track c) {
                                <option [value]="c">{{ titlecase(c) }}</option>
                            }
                        </optgroup>
                    }
                    <option [value]="CUSTOM_CLASSIFIER_KEY">+ Create new category…</option>
                </select>
                @if (classifierValue() === CUSTOM_CLASSIFIER_KEY) {
                    <input class="input df-mt-sm" type="text"
                           [value]="customClassifierValue()"
                           (input)="onCustomClassifierInput($event)"
                           placeholder="New category name"
                           autocomplete="off" spellcheck="false"/>
                }

                <label class="field-label df-mt">Trigger word</label>
                <div class="df-input-row">
                    <input class="input mono" type="text" formControlName="trigger_word"
                           autocomplete="off" spellcheck="false"
                           placeholder="e.g. mrlnstyle"/>
                    <button type="button" class="df-wand-btn"
                            (click)="generateTriggerFromName()"
                            [disabled]="!canGenerateTrigger()"
                            title="Generate from name"
                            aria-label="Generate trigger word from name">
                        <app-ico name="WandSparkles" [size]="14"/>
                    </button>
                </div>
                <div class="field-hint">Token baked into captions for LoRA activation.</div>

                <label class="field-label df-mt">Tags</label>
                <div class="df-chip-input"
                     [class.is-focused]="tagInputFocus()"
                     (click)="focusTagInput($event)">
                    @for (t of tags(); track $index) {
                        <span class="chip df-chip">
                            {{ t }}
                            <button type="button" class="df-chip-x" (click)="removeTag($index)"
                                    [attr.aria-label]="'Remove tag ' + t">
                                <app-ico name="X" [size]="10"/>
                            </button>
                        </span>
                    }
                    <input #tagInput class="df-chip-text" type="text"
                           [value]="tagDraft()"
                           (input)="onTagInput($event)"
                           (keydown)="onTagKeydown($event)"
                           (focus)="tagInputFocus.set(true)"
                           (blur)="onTagBlur()"
                           [placeholder]="tags().length === 0 ? 'Add tags — comma or Enter to confirm' : ''"
                           autocomplete="off" spellcheck="false"/>
                </div>

                <label class="field-label df-mt">Description</label>
                <textarea class="input df-textarea" rows="3" formControlName="description"
                          placeholder="Optional description for this dataset"></textarea>

                <label class="field-label df-mt">Notes / training hints</label>
                <textarea class="input df-textarea" rows="2" formControlName="notes"
                          placeholder="Internal notes — LR, rank, prompt tips, etc."></textarea>
            </form>
        </div>

        <div class="modal-foot">
            <button class="btn ghost" type="button" (click)="overlay.closeModal()">Cancel</button>
            <button class="btn primary" type="button"
                    [disabled]="form.invalid || nameInvalid() || submitting()"
                    (click)="submit()">
                @if (isEdit()) {
                    <app-ico name="Check" [size]="14"/>
                    {{ submitting() ? 'Saving…' : 'Save Changes' }}
                } @else {
                    <app-ico name="Plus" [size]="14"/>
                    {{ submitting() ? 'Creating…' : 'Create Dataset' }}
                }
            </button>
        </div>
    `,
    styles: [`
        .modal-title { font-size: 16px; font-weight: 700; margin-top: 2px; }
        .df-mt { margin-top: 12px; }
        .df-mt-sm { margin-top: 6px; }
        .df-textarea { min-height: 64px; resize: vertical; font-family: var(--font-sans); }
        .field-error {
            color: var(--color-danger);
            font-size: 11px;
            margin-top: 4px;
        }
        .field-hint {
            color: var(--color-text-muted);
            font-size: 11px;
            margin-top: 4px;
        }
        .input.mono { font-family: var(--font-mono, monospace); }

        .df-input-row {
            display: flex;
            gap: 6px;
            align-items: stretch;
        }
        .df-input-row .input { flex: 1 1 auto; min-width: 0; }
        .df-wand-btn {
            flex: 0 0 auto;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 34px;
            border: 1px solid var(--color-border, var(--color-border-subtle));
            background: var(--color-surface-input, var(--color-surface-low));
            color: var(--color-text-muted);
            border-radius: 6px;
            cursor: pointer;
            transition: color 0.12s ease, border-color 0.12s ease, background 0.12s ease;
        }
        .df-wand-btn:hover:not(:disabled) {
            color: var(--color-brand, #6b8afd);
            border-color: var(--color-brand, #6b8afd);
        }
        .df-wand-btn:disabled { opacity: 0.4; cursor: not-allowed; }

        .df-chip-input {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            min-height: 38px;
            padding: 6px 8px;
            border: 1px solid var(--color-border, var(--color-border-subtle));
            background: var(--color-surface-input, var(--color-surface-low));
            border-radius: 6px;
            cursor: text;
            transition: border-color 0.12s ease;
        }
        .df-chip-input.is-focused {
            border-color: var(--color-brand, #6b8afd);
            box-shadow: 0 0 0 1px var(--color-brand, #6b8afd);
        }
        .df-chip {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            padding: 2px 6px;
            font-size: 11px;
        }
        .df-chip-x {
            background: none;
            border: none;
            padding: 0;
            margin: 0 0 0 2px;
            display: inline-flex;
            align-items: center;
            color: inherit;
            cursor: pointer;
            opacity: 0.6;
        }
        .df-chip-x:hover { opacity: 1; }
        .df-chip-text {
            flex: 1 1 80px;
            min-width: 80px;
            border: none;
            outline: none;
            background: transparent;
            color: inherit;
            font: inherit;
            padding: 2px 0;
        }
    `],
})
export class DatasetFormModalComponent {
    private fb = inject(FormBuilder);
    private datasets = inject(DatasetStore);
    protected overlay = inject(OverlayStore);

    protected submitting = signal(false);
    protected tagInputFocus = signal(false);
    protected tagDraft = signal('');
    protected tags = signal<string[]>([]);

    /** Index of the next trigger strategy to apply on wand click. */
    private nextTriggerStrategy = signal(0);
    /** The last value the wand wrote, used to detect manual edits. */
    private lastGeneratedTrigger = signal('');

    /** Dataset id pulled from the topmost modal entry's data payload. */
    private datasetId = computed<string | null>(() => {
        const data = this.overlay.topModal()?.data as DatasetFormModalData | undefined;
        return data?.datasetId ?? null;
    });

    protected isEdit = computed<boolean>(() => this.datasetId() !== null);

    protected form = this.fb.nonNullable.group({
        name: ['', [nonEmptyTrimmed]],
        classifier: [''],
        trigger_word: [''],
        description: [''],
        notes: [''],
    });

    /** Template-accessible constants. */
    protected readonly standardClassifiers = STANDARD_CLASSIFIERS;
    protected readonly CUSTOM_CLASSIFIER_KEY = CUSTOM_CLASSIFIER_KEY;

    /**
     * Current value of the `classifier` form control, exposed as a signal
     * for the template (used to toggle the inline custom-entry input).
     * The form control itself drives the `<select>` via `formControlName`,
     * which handles the Angular @for + selected-option timing correctly.
     */
    protected classifierValue = toSignal(this.form.controls.classifier.valueChanges, {
        initialValue: this.form.controls.classifier.value,
    });
    /** Free-text value when the dropdown is in custom-entry mode. */
    protected customClassifierValue = signal('');

    /**
     * Categories observed on existing datasets that aren't in the standard
     * set — these become the "Previously used" optgroup, so a custom
     * category typed on one dataset becomes selectable on every other.
     */
    protected reusableClassifiers = computed<string[]>(() => {
        const std = new Set<string>([...STANDARD_CLASSIFIERS]);
        const seen = new Set<string>();
        const out: string[] = [];
        for (const d of this.datasets.entities()) {
            const c = (d.classifier ?? '').trim();
            if (!c) continue;
            if (std.has(c.toLowerCase())) continue;
            if (seen.has(c)) continue;
            seen.add(c);
            out.push(c);
        }
        return out.sort((a, b) => a.localeCompare(b));
    });

    /**
     * Wrap the name control's value stream as a signal so downstream computeds
     * re-evaluate on every keystroke.
     */
    private nameValue = toSignal(this.form.controls.name.valueChanges, {
        initialValue: this.form.controls.name.value ?? '',
    });

    protected nameInvalid = computed<boolean>(() => {
        const v = (this.nameValue() ?? '').trim();
        if (v.length === 0) return false;
        return NAME_FORBIDDEN.test(v);
    });

    protected canGenerateTrigger = computed<boolean>(() => {
        const v = (this.nameValue() ?? '').trim();
        return v.length > 0 && !NAME_FORBIDDEN.test(v);
    });

    /** Reactive lookup of the dataset being edited (null in create mode). */
    private editingDataset = computed(() => {
        const id = this.datasetId();
        return id ? this.datasets.byId(id)() : null;
    });

    private prefilled = signal(false);

    constructor() {
        // Prefill the form once when the dataset becomes available in
        // edit mode. Guarded so re-renders triggered by tags() / draft()
        // edits don't clobber the user's in-progress changes.
        effect(() => {
            if (!this.isEdit() || this.prefilled()) return;
            const ds = this.editingDataset();
            if (!ds) return;
            const saved = (ds.classifier ?? '').trim();
            const isStandard = (STANDARD_CLASSIFIERS as readonly string[]).includes(saved.toLowerCase());
            // Standards go through lowercased (matches option values); reusables
            // are kept verbatim — they match an option in the "Previously used"
            // group, which is sourced live from `datasets.entities()`.
            const classifier = saved ? (isStandard ? saved.toLowerCase() : saved) : '';
            this.form.patchValue({
                name: ds.name ?? '',
                classifier,
                trigger_word: ds.trigger_word ?? '',
                description: ds.description ?? '',
                notes: ds.notes ?? '',
            });
            this.customClassifierValue.set('');
            this.tags.set([...(ds.tags ?? [])]);
            this.prefilled.set(true);
        });

        // When the dropdown moves off the custom sentinel (e.g. the user
        // picks a different category), discard the in-progress custom draft.
        effect(() => {
            if (this.classifierValue() !== CUSTOM_CLASSIFIER_KEY && this.customClassifierValue() !== '') {
                this.customClassifierValue.set('');
            }
        });
    }

    protected titlecase(s: string): string {
        return s.charAt(0).toUpperCase() + s.slice(1);
    }

    // ── Category dropdown / inline custom entry ────────────────────────

    protected onCustomClassifierInput(ev: Event): void {
        this.customClassifierValue.set((ev.target as HTMLInputElement).value);
    }

    /** Resolve the persisted classifier from current UI state. */
    private resolveClassifier(): string {
        const sel = this.form.controls.classifier.value;
        if (sel === CUSTOM_CLASSIFIER_KEY) return this.customClassifierValue().trim();
        return sel;
    }

    /**
     * Derive a trigger word from the current name and write it into the
     * `trigger_word` field. The wand cycles through {@link TRIGGER_STRATEGIES}
     * on successive clicks — e.g. "911 Targa" yields, in order:
     * "911T4rga" → "9174rg4" → "911Targa" → "911targa" → "911T".
     *
     * The cycle resets to strategy 0 whenever the current field value
     * differs from the last value the wand wrote (i.e. the user edited
     * or cleared it manually), so a click after a manual edit always
     * gives the canonical "leet-light" result first.
     */
    protected generateTriggerFromName(): void {
        const raw = (this.form.controls.name.value ?? '').trim();
        if (!raw) return;

        const current = this.form.controls.trigger_word.value ?? '';
        const continuing = current !== '' && current === this.lastGeneratedTrigger();
        let idx = continuing ? this.nextTriggerStrategy() : 0;

        // Try strategies starting at `idx`; skip empty / duplicate results
        // (e.g. initials strategy on a single-word name). Loop at most N times.
        let trigger = '';
        const total = TRIGGER_STRATEGIES.length;
        for (let attempts = 0; attempts < total; attempts++) {
            const candidate = TRIGGER_STRATEGIES[idx](raw);
            idx = (idx + 1) % total;
            if (candidate && candidate !== current) {
                trigger = candidate;
                break;
            }
        }
        if (!trigger) return;

        this.form.controls.trigger_word.setValue(trigger);
        this.form.controls.trigger_word.markAsDirty();
        this.lastGeneratedTrigger.set(trigger);
        this.nextTriggerStrategy.set(idx);
    }

    // ── Tag chip input ─────────────────────────────────────────────────

    protected focusTagInput(ev: MouseEvent): void {
        const tgt = ev.target as HTMLElement;
        if (tgt.tagName === 'INPUT' || tgt.closest('.df-chip-x')) return;
        const input = (ev.currentTarget as HTMLElement).querySelector<HTMLInputElement>('.df-chip-text');
        input?.focus();
    }

    protected onTagInput(ev: Event): void {
        this.tagDraft.set((ev.target as HTMLInputElement).value);
    }

    protected onTagKeydown(ev: KeyboardEvent): void {
        if (ev.key === 'Enter' || ev.key === ',') {
            ev.preventDefault();
            this.commitDraftTag();
        } else if (ev.key === 'Backspace' && this.tagDraft().length === 0 && this.tags().length > 0) {
            this.removeTag(this.tags().length - 1);
        }
    }

    protected onTagBlur(): void {
        // Commit on blur so a user who tabbed away doesn't lose their draft.
        this.commitDraftTag();
        this.tagInputFocus.set(false);
    }

    private commitDraftTag(): void {
        const raw = this.tagDraft().trim();
        if (!raw) return;
        // Split on comma in case the user pasted "a, b, c".
        const parts = raw.split(',').map(s => s.trim()).filter(Boolean);
        if (parts.length === 0) return;
        const existing = new Set(this.tags());
        const next = [...this.tags()];
        for (const p of parts) {
            if (!existing.has(p)) {
                next.push(p);
                existing.add(p);
            }
        }
        this.tags.set(next);
        this.tagDraft.set('');
    }

    protected removeTag(index: number): void {
        const next = this.tags().slice();
        next.splice(index, 1);
        this.tags.set(next);
    }

    // ── Submit ─────────────────────────────────────────────────────────

    async submit(): Promise<void> {
        if (this.form.invalid || this.nameInvalid() || this.submitting()) return;

        // Flush any uncommitted tag draft before reading the form.
        if (this.tagDraft().trim()) this.commitDraftTag();

        const { name, description, trigger_word, notes } = this.form.getRawValue();
        const classifier = this.resolveClassifier();
        const extra = {
            trigger_word: (trigger_word ?? '').trim(),
            tags: this.tags(),
            notes: notes ?? '',
        };

        this.submitting.set(true);
        try {
            if (this.isEdit()) {
                const id = this.datasetId()!;
                await this.datasets.updateDataset(id, {
                    name: (name ?? '').trim(),
                    description: description ?? '',
                    classifier,
                    trigger_word: extra.trigger_word,
                    tags: extra.tags,
                    notes: extra.notes,
                });
                this.overlay.closeModal();
            } else {
                const created = await this.datasets.createDataset(
                    (name ?? '').trim(),
                    description ?? '',
                    classifier,
                    extra,
                );
                if (created) this.overlay.closeModal();
            }
        } finally {
            this.submitting.set(false);
        }
    }
}
