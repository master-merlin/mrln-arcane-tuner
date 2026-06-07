import { ChangeDetectionStrategy, Component, input } from '@angular/core';

/**
 * Coverage tier for a dataset-level pill. Drives the green/amber/red/grey
 * color in the Library (and anywhere a pill reflects % completeness rather
 * than a binary file-level yes/no).
 */
export type PillLevel = 'full' | 'mid' | 'low' | 'none';

export interface StatePillsState {
  harmonized?: boolean;
  captioned?: boolean;
  masked?: boolean;
  /**
   * Optional coverage tiers. When present, each pill is colored by completeness
   * (full ≥90% → green, mid ≥50% → amber, low >0% → red, none → grey) instead of
   * the binary per-letter hue. Used by dataset-level surfaces (Library card,
   * training-form rows). Omit for file-level surfaces (Grid, analyze) which stay
   * binary letter-colored.
   */
  levels?: {
    harmonized?: PillLevel;
    captioned?: PillLevel;
    masked?: PillLevel;
  };
  /** Optional native tooltips per pill — used to surface coverage details. */
  titles?: {
    harmonized?: string;
    captioned?: string;
    masked?: string;
  };
}

/** Ratio (0–1) → coverage tier. ≥90% full, ≥50% mid, >0% low, else none. */
export function levelFromRatio(ratio: number): PillLevel {
  if (!(ratio > 0)) return 'none';
  if (ratio >= 0.9) return 'full';
  if (ratio >= 0.5) return 'mid';
  return 'low';
}

/**
 * Build the dataset-level (Library) pill state from raw coverage counts. Shared
 * by the dataset Library card and the training-form rows so the two can never
 * drift. `harmonizationScore` is the backend's 0–1 aggregate ratio.
 */
export function datasetStatePills(input: {
  total: number;
  captioned: number;
  masked: number;
  harmonizationScore: number;
}): StatePillsState {
  const { total, captioned, masked, harmonizationScore } = input;
  const capRatio = total > 0 ? captioned / total : 0;
  const maskRatio = total > 0 ? masked / total : 0;
  const harmonized = Math.round(harmonizationScore * total);
  const pct = (n: number) => (total > 0 ? Math.round((n / total) * 100) : 0);
  const fmt = (label: string, n: number, percent: number) =>
    total > 0 ? `${label} ${n}/${total} files (${percent}%)` : `${label}: no images yet`;
  return {
    harmonized: harmonizationScore > 0,
    captioned: capRatio > 0,
    masked: maskRatio > 0,
    levels: {
      harmonized: levelFromRatio(harmonizationScore),
      captioned: levelFromRatio(capRatio),
      masked: levelFromRatio(maskRatio),
    },
    titles: {
      harmonized: fmt('Harmonized', harmonized, Math.round(harmonizationScore * 100)),
      captioned: fmt('Captioned', captioned, pct(captioned)),
      masked: fmt('Masked', masked, pct(masked)),
    },
  };
}

/** H / C / M readiness trio — wraps `.state-pills` + `.state-pill`. */
@Component({
  selector: 'app-state-pills',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  host: { style: 'display: inline-flex;' },
  template: `
    <span class="state-pills">
      <span
        class="state-pill H"
        data-testid="state-pill-harmonized"
        [class.on]="isOn('harmonized')"
        [class.lvl-full]="state().levels?.harmonized === 'full'"
        [class.lvl-mid]="state().levels?.harmonized === 'mid'"
        [class.lvl-low]="state().levels?.harmonized === 'low'"
        [attr.title]="state().titles?.harmonized ?? null"
        >H</span
      >
      <span
        class="state-pill C"
        data-testid="state-pill-captioned"
        [class.on]="isOn('captioned')"
        [class.lvl-full]="state().levels?.captioned === 'full'"
        [class.lvl-mid]="state().levels?.captioned === 'mid'"
        [class.lvl-low]="state().levels?.captioned === 'low'"
        [attr.title]="state().titles?.captioned ?? null"
        >C</span
      >
      <span
        class="state-pill M"
        data-testid="state-pill-masked"
        [class.on]="isOn('masked')"
        [class.lvl-full]="state().levels?.masked === 'full'"
        [class.lvl-mid]="state().levels?.masked === 'mid'"
        [class.lvl-low]="state().levels?.masked === 'low'"
        [attr.title]="state().titles?.masked ?? null"
        >M</span
      >
    </span>
  `,
})
export class StatePillsComponent {
  state = input.required<StatePillsState>();

  /** A pill is "on" when its coverage tier is non-none, or — in binary mode
   *  (no levels) — when its boolean flag is set. */
  protected isOn(key: 'harmonized' | 'captioned' | 'masked'): boolean {
    const s = this.state();
    const lv = s.levels?.[key];
    if (lv !== undefined) return lv !== 'none';
    return !!s[key];
  }
}
