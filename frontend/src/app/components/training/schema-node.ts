/**
 * One node of a plugin's training JSON schema.
 *
 * This is a custom JSON-Schema dialect: standard keywords (`type`, `title`,
 * `description`, `default`, `enum`, `properties`, `items`, `required`, `$ref`,
 * `$defs`/`definitions`) plus UI extensions the dynamic form renderer reads
 * (`input_type`, `ui_type`, `display`, `inline_group`, `group`, `depends_on`,
 * `disabled_if`/`hidden_if`, `backend_map`, `options_labels`/`enum_labels`,
 * `hide_unsupported`, `hidden`, numeric `min`/`max`/`step`). Recursive via
 * `items` / `properties` / `$defs`. Every field is optional — a node only
 * carries what its kind needs.
 */
export interface SchemaNode {
  type?: string;
  title?: string;
  description?: string;
  default?: unknown;
  required?: string[];

  // Enum / select options
  enum?: string[];
  enum_labels?: Record<string, string>;
  options_labels?: Record<string, string>;

  // Numeric constraints
  min?: number;
  max?: number;
  step?: number;
  /** Minimum row count for arrays (pydantic `min_length` → JSON-Schema `minItems`). */
  minItems?: number;

  // UI / rendering extensions
  input_type?: string;
  ui_type?: string;
  display?: string;
  readOnly?: boolean;
  hidden?: boolean;
  hide_unsupported?: boolean;
  inline_group?: string;
  group?: string;

  // Conditional visibility / enablement
  depends_on?: string;
  disabled_if?: Record<string, unknown>;
  hidden_if?: Record<string, unknown>;
  /** Show this field only for video models (gated on the model's is_video
   *  capability). Used by per-dataset video knobs (e.g. num_frames). */
  video_only?: boolean;
  /** backend name → schemes that backend supports (e.g. quantization). Values
   *  are usually a plain scheme list, but some schemas wrap them as `{schemes}`. */
  backend_map?: Record<string, string[] | { schemes?: string[] }>;

  // Composition
  items?: SchemaNode;
  properties?: Record<string, SchemaNode>;
  $ref?: string;
  $defs?: Record<string, SchemaNode>;
  definitions?: Record<string, SchemaNode>;
}

/** One renderable property: its key plus its (ref-resolved) schema node. */
export interface SchemaProp {
  key: string;
  schema: SchemaNode;
}
