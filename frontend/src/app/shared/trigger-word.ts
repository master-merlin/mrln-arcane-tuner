/**
 * Trigger-word generation — the single source of truth shared by the dataset
 * form modal and the training config form. Both surfaces offer a "magic wand"
 * that derives a trigger word from a name, cycling through phrasings on repeat
 * clicks. Keeping the strategies here prevents the two call sites from drifting.
 */

const LEET_LIGHT: Record<string, string> = {
  a: '4', A: '4', e: '3', E: '3', i: '1', I: '1', o: '0', O: '0',
};
const LEET_FULL: Record<string, string> = {
  ...LEET_LIGHT,
  s: '5', S: '5', t: '7', T: '7',
};

/** Strip every non-alphanumeric separator, preserving case. */
export function stripSeparators(s: string): string {
  return s.replace(/[^A-Za-z0-9]+/g, '');
}

/**
 * Trigger-word generation strategies. Each takes the raw (untrimmed) name and
 * returns a candidate. Empty results mean "not applicable for this input" and
 * the caller should skip to the next strategy.
 *
 * The wand button cycles through these on repeat clicks, giving the user
 * alternative phrasings without forcing them to invent one.
 */
export const TRIGGER_STRATEGIES: ReadonlyArray<(raw: string) => string> = [
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

/**
 * Produce the next trigger-word candidate from `raw`, starting at strategy
 * `startIndex` and cycling forward, skipping empty results and duplicates of
 * `current`. Returns the chosen candidate plus the index the next click should
 * resume from, or `null` when no usable candidate exists.
 *
 * @param raw         source name (e.g. a dataset name); untrimmed is fine
 * @param current     the field's current value, to avoid re-emitting it
 * @param startIndex  strategy index to begin at (use 0 to restart the cycle)
 */
export function nextTriggerWord(
  raw: string,
  current: string,
  startIndex: number,
): { trigger: string; nextIndex: number } | null {
  const trimmed = (raw ?? '').trim();
  if (!trimmed) return null;

  const total = TRIGGER_STRATEGIES.length;
  let idx = ((startIndex % total) + total) % total;

  for (let attempts = 0; attempts < total; attempts++) {
    const candidate = TRIGGER_STRATEGIES[idx](trimmed);
    idx = (idx + 1) % total;
    if (candidate && candidate !== current) {
      return { trigger: candidate, nextIndex: idx };
    }
  }
  return null;
}
