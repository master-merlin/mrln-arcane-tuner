/**
 * Single source of truth for all 12 pipeline operations + the Crop tab.
 * The PipelineBlock.type strings here MUST match the backend's
 * pipeline-op identifiers — verified against the legacy editor at
 * [image-editor-modal.ts:960-962, 2078].
 */

export type OperationKind =
    | 'denoise'
    | 'face_restore'
    | 'white_balance'
    | 'curves'
    | 'lut'
    | 'color_match'
    | 'hsl_selective'
    | 'color_tone'  // one panel → two pipeline rows (hue_saturation + contrast)
    | 'vignette'
    | 'lens'
    | 'sharpen'
    | 'upscale';

/** Pseudo-kind for the Crop tab. Not a pipeline op (destructive, separate flow). */
export type TabKind = OperationKind | 'crop';

export type TabGroup = 'adjust' | 'ai';

export interface TabDef {
    kind: TabKind;
    label: string;
    icon: string;       // @lucide/angular icon name
    group: TabGroup;
}

/** Order in the left-pane tab bar, grouped. */
export const TAB_DEFS: ReadonlyArray<TabDef> = [
    // Adjust
    { kind: 'crop',           label: 'Crop',         icon: 'Square',   group: 'adjust' },
    { kind: 'white_balance',  label: 'White Balance',icon: 'Thermometer', group: 'adjust' },
    { kind: 'curves',         label: 'Curves',       icon: 'Activity', group: 'adjust' },
    { kind: 'color_tone',     label: 'Color & Tone', icon: 'Wand2',    group: 'adjust' },
    { kind: 'hsl_selective',  label: 'HSL',          icon: 'Palette',  group: 'adjust' },
    { kind: 'sharpen',        label: 'Sharpen',      icon: 'Zap',      group: 'adjust' },
    { kind: 'vignette',       label: 'Vignette',     icon: 'Aperture', group: 'adjust' },
    { kind: 'lens',           label: 'Lens',         icon: 'Compass',  group: 'adjust' },
    { kind: 'lut',            label: 'CUBE LUT',     icon: 'Files',    group: 'adjust' },
    { kind: 'color_match',    label: 'Color Match',  icon: 'Pipette',  group: 'adjust' },
    // AI Models
    { kind: 'denoise',        label: 'Denoise',      icon: 'Sparkles', group: 'ai' },
    { kind: 'face_restore',   label: 'Face Restore', icon: 'Smile',    group: 'ai' },
    { kind: 'upscale',        label: 'Upscale',      icon: 'ZoomIn',   group: 'ai' },
];

/**
 * Backend `OPERATION_ORDER` — verified at
 * [image-editor-modal.ts:960-962]. Color Match is NOT in this list
 * (backend always applies it first; not user-reorderable).
 */
export const PIPELINE_ORDER: ReadonlyArray<OperationKind> = [
    'denoise',
    'face_restore',
    'white_balance',
    'curves',
    'lut',
    'hsl_selective',
    'color_tone',  // expands to hue_saturation + contrast at block-emit time
    'vignette',
    'lens',
    'sharpen',
    'upscale',
];

/**
 * Map a frontend OperationKind to one or more backend pipeline-block
 * `type` strings. `color_tone` expands to two blocks; `lut` is `cube_lut`;
 * `sharpen` is `sharpening` (common typo trap — see spec); `lens` is
 * `lens_correction`.
 */
export const BACKEND_TYPE_FOR: Record<OperationKind, string | string[]> = {
    denoise:        'denoise',
    face_restore:   'face_restore',
    white_balance:  'white_balance',
    curves:         'curves',
    lut:            'cube_lut',
    color_match:    'color_match',
    hsl_selective:  'hsl_selective',
    color_tone:     ['hue_saturation', 'contrast'],
    vignette:       'vignette',
    lens:           'lens_correction',
    sharpen:        'sharpening',
    upscale:        'upscale',
};

// ── Per-op param shapes ─────────────────────────────────────────────────
export interface WBParams { temperature: number; tint: number; }
export interface ColorToneParams { hue_shift: number; saturation: number; contrast: number; }
export interface SharpenParams {
    method: 'unsharp' | 'kernel' | 'high_pass';
    radius: number; amount: number; threshold: number; strength: number;
}
export interface VignetteParams {
    amount: number; midpoint: number; feather: number;
    shape: 'circular' | 'rectangular'; apply_before_lut: boolean;
}
export interface LensParams {
    barrel: number; v_keystone: number; h_keystone: number; auto_crop: boolean;
}
export interface LutEntry { file: string; strength: number; enabled: boolean; }
export interface LutParams { luts: LutEntry[]; tetrahedral: boolean; }
export interface ColorMatchParams {
    reference_path: string | null;
    method: 'cdf' | 'wavelet';
    strength: number;
}
export interface RestoreParams {
    folder: string; model: string | null; strength: number; tile_size: number;
    face_only?: boolean;  // face_restore only
}
export interface UpscaleParams {
    folder: string; model: string | null; tile_size: number;
    target_scale: 1 | 2 | 4 | 8;
    resize_method: 'lanczos' | 'bicubic' | 'bilinear' | 'nearest';
}
export interface CurvePoint { x: number; y: number; }
export interface CurvesParams {
    master: CurvePoint[]; r: CurvePoint[]; g: CurvePoint[]; b: CurvePoint[];
}
export interface HslRangeAdj { hue_shift: number; saturation: number; luminance: number; }
export interface HslParams { [range: string]: HslRangeAdj; }

const IDENTITY_CURVE: CurvePoint[] = [{ x: 0, y: 0 }, { x: 255, y: 255 }];

export const DEFAULT_PARAMS = {
    white_balance:  { temperature: 6500, tint: 0 } satisfies WBParams,
    curves:         { master: [...IDENTITY_CURVE], r: [...IDENTITY_CURVE], g: [...IDENTITY_CURVE], b: [...IDENTITY_CURVE] } satisfies CurvesParams,
    lut:            { luts: [], tetrahedral: true } satisfies LutParams,
    color_match:    { reference_path: null, method: 'cdf', strength: 0.5 } satisfies ColorMatchParams,
    hsl_selective:  {} satisfies HslParams,
    color_tone:     { hue_shift: 0, saturation: 1.0, contrast: 1.0 } satisfies ColorToneParams,
    vignette:       { amount: 0, midpoint: 0.5, feather: 0.5, shape: 'circular', apply_before_lut: false } satisfies VignetteParams,
    lens:           { barrel: 0, v_keystone: 0, h_keystone: 0, auto_crop: true } satisfies LensParams,
    sharpen:        { method: 'unsharp', radius: 2, amount: 150, threshold: 3, strength: 1.0 } satisfies SharpenParams,
    denoise:        { folder: 'models/restore', model: null, strength: 0.6, tile_size: 512 } satisfies RestoreParams,
    face_restore:   { folder: 'models/restore', model: null, strength: 0.6, tile_size: 512, face_only: true } satisfies RestoreParams,
    upscale:        { folder: 'models/upscale', model: null, tile_size: 512, target_scale: 2, resize_method: 'lanczos' } satisfies UpscaleParams,
} as const;
