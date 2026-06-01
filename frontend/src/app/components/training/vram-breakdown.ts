import { VRAMReport } from '../../services/system.service';

/** One segment of the VRAM breakdown bar / legend. */
export interface BreakdownPart {
  key: string;
  label: string;
  color: string;
  mb: number;
  /** Always shown in the legend even when mb is 0 (Model, Headroom). */
  always: boolean;
}

/**
 * Canonical VRAM breakdown parts derived from a {@link VRAMReport}, in render
 * order, with the SINGLE source-of-truth color mapping shared by the Live
 * Estimate rail and the VRAM Budget card so the two surfaces stay visually
 * consistent. `Headroom` = max(0, available - peak), rendered transparent so a
 * bar background shows through. Never hardcodes example numbers.
 */
export function vramBreakdownParts(r: VRAMReport | null | undefined): BreakdownPart[] {
  if (!r) return [];
  const headroom = Math.max(0, r.available_mb - r.peak_mb);
  return [
    { key: 'model', label: 'Model', color: 'var(--color-brand)', mb: r.model_weights_mb, always: true },
    { key: 'adapters', label: 'Adapters', color: 'var(--color-chart-lr, #38bdf8)', mb: r.lora_adapters_mb, always: false },
    { key: 'optimizer', label: 'Optimizer', color: 'var(--color-violet, #8b5cf6)', mb: r.optimizer_states_mb, always: false },
    { key: 'gradients', label: 'Gradients', color: 'var(--color-warning, #fbbf24)', mb: r.gradients_mb, always: false },
    { key: 'activations', label: 'Activations', color: 'var(--color-success, #34d399)', mb: r.activations_mb, always: false },
    { key: 'overhead', label: 'Overhead', color: 'var(--color-text-subtle, #6b7280)', mb: r.overhead_mb, always: false },
    { key: 'headroom', label: 'Headroom', color: 'transparent', mb: headroom, always: true },
  ];
}
