import { ChangeDetectionStrategy, Component, input } from '@angular/core';

/** One key/value row shown in the template info card. */
export interface TemplateInfoRow {
  key: string;
  value: string;
}

/**
 * Reusable "selected template" info card — a colored dot, the template name,
 * and an auto-fitting grid of key/value rows pulled from the template config.
 *
 * Extracted from the Projects → Quick Train panel so the training template
 * selector can render the identical card. The dot color is an input so each
 * host can tint it (Quick Train uses the domain tone; the training selector
 * uses the brand color, which is the default).
 */
@Component({
  selector: 'app-template-info-card',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="ti-card">
      <span class="ti-dot" [style.background]="dotColor()"></span>
      <div class="ti-text">
        <div class="ti-name">{{ name() }}</div>
        @if (rows().length) {
          <dl class="ti-kv">
            @for (row of rows(); track row.key) {
              <div class="ti-kv-row">
                <dt>{{ row.key }}</dt>
                <dd>{{ row.value }}</dd>
              </div>
            }
          </dl>
        }
      </div>
    </div>
  `,
  styles: [`
    .ti-card {
      display: flex; align-items: flex-start; gap: 10px;
      padding: 12px 14px;
      background: var(--color-surface-mid);
      border: 1px solid var(--color-border-subtle);
      border-radius: var(--radius-theme-md);
    }
    .ti-dot { width: 8px; height: 8px; border-radius: 2px; flex-shrink: 0; margin-top: 5px; }
    .ti-text { display: flex; flex-direction: column; gap: 8px; min-width: 0; flex: 1; }
    .ti-name { font-size: 13px; font-weight: 600; color: var(--color-text-primary); }
    .ti-kv {
      margin: 0;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 8px 32px;
    }
    .ti-kv-row { display: flex; align-items: baseline; justify-content: flex-start; gap: 7px; min-width: 0; }
    .ti-kv dt {
      font-size: 10px; font-weight: 600;
      text-transform: uppercase; letter-spacing: 0.06em;
      color: var(--color-text-muted);
      white-space: nowrap;
    }
    .ti-kv dd {
      margin: 0;
      font-family: var(--font-mono); font-size: 11.5px; font-weight: 700;
      color: var(--color-text-secondary);
      overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    }
  `],
})
export class TemplateInfoCardComponent {
  name = input<string>('');
  rows = input<TemplateInfoRow[]>([]);
  /** Dot tint; defaults to the brand color. */
  dotColor = input<string>('var(--color-brand)');
}
