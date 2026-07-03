/**
 * Ideogram 4 structured-caption schema — TypeScript mirror of
 * backend/app/core/captioning/formats/schema/ideogram4.py
 *
 * Pure functions only — no I/O, no Angular deps.
 * Key-order and serialization match the Python backend exactly so that
 * serialize(normalize(x)) === python_serialize(python_normalize(x)).
 */

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

export const PHOTO_MEDIUM = 'photograph';

export const CANONICAL_MEDIUMS = [
  'photograph',
  'illustration',
  '3d_render',
  'painting',
  'graphic_design',
] as const;

export const MAX_IMAGE_PALETTE = 16;
export const MAX_ELEMENT_PALETTE = 5;
export const BBOX_MAX = 1000;

// ---------------------------------------------------------------------------
// Medium aliases (mirrors _MEDIUM_ALIASES in Python)
// ---------------------------------------------------------------------------

const MEDIUM_ALIASES: Record<string, string> = {
  photo: 'photograph',
  photograph: 'photograph',
  photography: 'photograph',
  illustration: 'illustration',
  drawing: 'illustration',
  '3d': '3d_render',
  '3d_render': '3d_render',
  '3drender': '3d_render',
  render: '3d_render',
  cgi: '3d_render',
  painting: 'painting',
  oil: 'painting',
  'oil painting': 'painting',
  graphic_design: 'graphic_design',
  'graphic design': 'graphic_design',
};

// ---------------------------------------------------------------------------
// Interfaces
// ---------------------------------------------------------------------------

export interface IdeogramStyle {
  aesthetics: string;
  lighting: string;
  /** Photo branch only */
  photo?: string;
  /** Non-photo branch only */
  art_style?: string;
  medium: string;
  color_palette: string[];
}

export interface IdeogramElement {
  type: 'obj' | 'text';
  bbox?: number[];
  /** Text elements only */
  text?: string;
  desc: string;
  color_palette: string[];
}

export interface IdeogramCaption {
  high_level_description: string;
  style_description: IdeogramStyle;
  compositional_deconstruction: {
    background: string;
    elements: IdeogramElement[];
  };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Canonicalize a medium string — mirrors Python canon_medium().
 */
export function canonMedium(m: string): string {
  // mirrors: key = (m or "").strip().lower().rstrip(".").strip()
  let key = (m ?? '').trim().toLowerCase().replace(/\.*$/, '').trim();
  if (key in MEDIUM_ALIASES) {
    return MEDIUM_ALIASES[key];
  }
  const key2 = key.replace(/ /g, '_');
  return MEDIUM_ALIASES[key2] ?? (key2 || PHOTO_MEDIUM);
}

/**
 * Normalize a hex colour string — mirrors Python normalize_color().
 * Returns '#RRGGBB' (uppercase) or null.
 */
export function normalizeColor(c: unknown): string | null {
  if (typeof c !== 'string') return null;
  let s = c.trim().replace(/^#/, '');
  // Expand 3-digit hex
  if (/^[0-9a-fA-F]{3}$/.test(s)) {
    s = s[0] + s[0] + s[1] + s[1] + s[2] + s[2];
  }
  if (/^[0-9a-fA-F]{6}$/.test(s)) {
    return '#' + s.toUpperCase();
  }
  return null;
}

/** Build a deduplicated, capped, normalised palette. */
function buildPalette(colors: unknown[], cap: number): string[] {
  const out: string[] = [];
  for (const c of colors ?? []) {
    const nc = normalizeColor(c);
    if (nc && !out.includes(nc)) {
      out.push(nc);
    }
    if (out.length >= cap) break;
  }
  return out;
}

/** Clamp/round a bbox to [0, BBOX_MAX] — mirrors Python _clamp_bbox(). */
function clampBbox(bbox: unknown): number[] | null {
  if (!Array.isArray(bbox) || bbox.length !== 4) return null;
  try {
    const vals = (bbox as unknown[]).map((v) => {
      const n = parseFloat(String(v));
      if (!isFinite(n)) throw new Error('non-finite');
      return Math.max(0, Math.min(BBOX_MAX, Math.round(n)));
    });
    return vals;
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// Normalise sub-structures
// ---------------------------------------------------------------------------

function normalizeStyle(style: Record<string, unknown>): IdeogramStyle {
  const s = style ?? {};
  const medium = canonMedium(String(s['medium'] ?? PHOTO_MEDIUM));

  // render field: prefer 'photo', fall back to 'art_style'
  const render: string =
    s['photo'] != null
      ? String(s['photo'])
      : s['art_style'] != null
        ? String(s['art_style'])
        : '';

  const aesthetics = String(s['aesthetics'] ?? '');
  const lighting = String(s['lighting'] ?? '');
  const palette = buildPalette(
    Array.isArray(s['color_palette']) ? (s['color_palette'] as unknown[]) : [],
    MAX_IMAGE_PALETTE,
  );

  if (medium === PHOTO_MEDIUM) {
    // Photo branch key order: aesthetics, lighting, photo, medium, color_palette
    return {
      aesthetics,
      lighting,
      photo: render,
      medium,
      color_palette: palette,
    };
  } else {
    // Non-photo branch key order: aesthetics, lighting, medium, art_style, color_palette
    return {
      aesthetics,
      lighting,
      medium,
      art_style: render,
      color_palette: palette,
    };
  }
}

function normalizeElement(el: Record<string, unknown>): IdeogramElement {
  const etype: 'text' | 'obj' = el['type'] === 'text' ? 'text' : 'obj';
  const bbox = clampBbox(el['bbox']);
  const palette = buildPalette(
    Array.isArray(el['color_palette']) ? (el['color_palette'] as unknown[]) : [],
    MAX_ELEMENT_PALETTE,
  );
  const desc = String(el['desc'] ?? '');

  if (etype === 'text') {
    // Text element key order: type, bbox, text, desc, color_palette
    const result: IdeogramElement = {
      type: 'text',
      ...(bbox != null && { bbox }),
      text: String(el['text'] ?? ''),
      desc,
      color_palette: palette,
    };
    return result;
  } else {
    // Obj element key order: type, bbox, desc, color_palette
    const result: IdeogramElement = {
      type: 'obj',
      ...(bbox != null && { bbox }),
      desc,
      color_palette: palette,
    };
    return result;
  }
}

function normalizeDeconstruction(dec: Record<string, unknown>): {
  background: string;
  elements: IdeogramElement[];
} {
  const d = dec ?? {};
  const elements: unknown[] = Array.isArray(d['elements']) ? (d['elements'] as unknown[]) : [];
  return {
    background: String(d['background'] ?? ''),
    elements: elements
      .filter((e): e is Record<string, unknown> => typeof e === 'object' && e !== null)
      .map((e) => normalizeElement(e)),
  };
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Normalise a raw (possibly dirty) ideogram4 document into a canonical shape.
 * Mirrors Python normalize().
 *
 * Key insertion order is intentional — JS preserves string-key insertion order,
 * so JSON.stringify produces the same compact string as Python json.dumps with
 * the same key order.
 */
export function normalize(data: Record<string, unknown>): IdeogramCaption {
  const d = data ?? {};
  return {
    high_level_description: String(d['high_level_description'] ?? ''),
    style_description: normalizeStyle(
      (d['style_description'] as Record<string, unknown>) ?? {},
    ),
    compositional_deconstruction: normalizeDeconstruction(
      (d['compositional_deconstruction'] as Record<string, unknown>) ?? {},
    ),
  };
}

/**
 * Compact JSON serialization — mirrors Python json.dumps(ensure_ascii=False, separators=(",",":")).
 * JS JSON.stringify already uses compact separators and does NOT escape non-ASCII.
 */
export function serialize(data: IdeogramCaption): string {
  return JSON.stringify(data);
}

/**
 * Parse ideogram4 JSON text (lenient: strips ```json fences, falls back to
 * first balanced {...} block).  Mirrors Python parse().
 */
export function parse(text: string): Record<string, unknown> | null {
  if (!text) return null;
  let s = text.trim();

  // Strip ```json fences
  const fence = s.match(/```(?:json)?\s*(\{[\s\S]*\})\s*```/);
  if (fence) {
    s = fence[1];
  }

  try {
    const obj = JSON.parse(s);
    return typeof obj === 'object' && obj !== null && !Array.isArray(obj)
      ? (obj as Record<string, unknown>)
      : null;
  } catch {
    // Fallback: first balanced {...} block
    const start = s.indexOf('{');
    const end = s.lastIndexOf('}');
    if (start !== -1 && end > start) {
      try {
        const obj = JSON.parse(s.slice(start, end + 1));
        return typeof obj === 'object' && obj !== null && !Array.isArray(obj)
          ? (obj as Record<string, unknown>)
          : null;
      } catch {
        return null;
      }
    }
    return null;
  }
}

/**
 * Returns true if the text is valid JSON with a compositional_deconstruction key.
 * Mirrors Python detect().
 */
export function detect(text: string): boolean {
  const obj = parse(text);
  return obj != null && 'compositional_deconstruction' in obj;
}

/**
 * Wrap raw text as high_level_description into a normalized skeleton document.
 * Mirrors Python skeleton().
 */
export function skeleton(rawText: string): IdeogramCaption {
  return normalize({
    high_level_description: (rawText ?? '').trim(),
    style_description: {},
    compositional_deconstruction: { background: '', elements: [] },
  });
}
