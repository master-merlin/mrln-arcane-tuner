import {
    ChangeDetectionStrategy,
    Component,
    ElementRef,
    OnDestroy,
    afterNextRender,
    effect,
    input,
    output,
    viewChild,
} from '@angular/core';
import { EditorView, keymap, lineNumbers } from '@codemirror/view';
import { history, historyKeymap, defaultKeymap, indentWithTab } from '@codemirror/commands';
import { bracketMatching, indentOnInput } from '@codemirror/language';
import { json, jsonParseLinter } from '@codemirror/lang-json';
import { linter, lintGutter } from '@codemirror/lint';
import { oneDark } from '@codemirror/theme-one-dark';

/**
 * Reusable CodeMirror 6 editor for JSON. Syntax-highlighted (one-dark),
 * linted (invalid JSON underlined + reported via `validChange`), editable.
 *
 * Two-way-ish: `value` pushes external content in; `valueChange` emits on
 * every edit; `validChange` emits JSON validity so a host can gate a Save
 * button. Kept dependency-light — code editors read fine as dark in both
 * app themes, so no custom theme is wired.
 */
@Component({
    selector: 'app-json-editor',
    standalone: true,
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `<div class="je-host" #host></div>`,
    styles: [`
        :host { display: block; height: 100%; }
        .je-host { height: 100%; }
    `],
})
export class JsonEditorComponent implements OnDestroy {
    readonly value = input<string>('');
    readonly readOnly = input<boolean>(false);
    readonly valueChange = output<string>();
    readonly validChange = output<boolean>();

    private host = viewChild.required<ElementRef<HTMLDivElement>>('host');
    private view?: EditorView;
    private applyingExternal = false;

    constructor() {
        afterNextRender(() => this.init());
        // Push external `value` changes into the editor without echoing back.
        effect(() => {
            const v = this.value();
            const view = this.view;
            if (!view || v === view.state.doc.toString()) return;
            this.applyingExternal = true;
            view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: v } });
            this.applyingExternal = false;
            this.emitValidity(v);
        });
    }

    private init(): void {
        const listener = EditorView.updateListener.of(u => {
            if (u.docChanged && !this.applyingExternal) {
                const text = u.state.doc.toString();
                this.valueChange.emit(text);
                this.emitValidity(text);
            }
        });
        this.view = new EditorView({
            doc: this.value(),
            parent: this.host().nativeElement,
            extensions: [
                lineNumbers(),
                history(),
                bracketMatching(),
                indentOnInput(),
                json(),
                linter(jsonParseLinter()),
                lintGutter(),
                keymap.of([...defaultKeymap, ...historyKeymap, indentWithTab]),
                oneDark,
                // Wrap long lines so a single long value never forces horizontal
                // scrolling — the line-number gutter still marks each logical line.
                EditorView.lineWrapping,
                // Height + scrolling MUST be set via the CM theme, not component
                // CSS: CodeMirror builds .cm-editor/.cm-scroller dynamically, so
                // Angular's emulated-encapsulation attribute never lands on them
                // and scoped `.cm-*` rules silently don't apply. The editor fills
                // its host (a fixed-height container) and the scroller handles
                // vertical overflow.
                EditorView.theme({
                    '&': { height: '100%', borderRadius: 'var(--radius-theme-md)' },
                    '.cm-scroller': {
                        overflow: 'auto',
                        fontFamily: 'var(--font-mono)',
                        fontSize: '12.5px',
                    },
                }),
                EditorView.editable.of(!this.readOnly()),
                listener,
            ],
        });
        this.emitValidity(this.value());
    }

    private emitValidity(text: string): void {
        let valid = true;
        try { JSON.parse(text); } catch { valid = false; }
        this.validChange.emit(valid);
    }

    ngOnDestroy(): void {
        this.view?.destroy();
    }
}
