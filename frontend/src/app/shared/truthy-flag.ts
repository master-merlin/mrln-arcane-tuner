/**
 * Read a boolean flag defensively — the single source of truth for both the
 * training form's submit-time gating and the adaptive-targeting card's knob
 * normalization.
 *
 * Checkbox controls normally hold a real boolean, but a value round-tripped
 * through a template, a job config or a JSON editor arrives as the STRING
 * `"false"` — which is truthy, and would flip a gate the wrong way. A strict
 * `=== true` has the mirror-image bug: it maps a round-tripped `"true"` to
 * false. Both call sites must agree, so the predicate lives here.
 */
export function isTruthyFlag(v: unknown): boolean {
  if (typeof v === 'string') {
    const s = v.trim().toLowerCase();
    return s !== '' && s !== 'false' && s !== '0';
  }
  return !!v;
}
