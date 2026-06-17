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
  /** Pydantic emits `Optional[T]` / `T | None` as `anyOf: [{type:T}, {type:null}]`
   *  with no scalar `type` on the node. The renderer keys off a scalar `type`,
   *  so these must be collapsed via {@link collapseNullableUnion} at resolve time. */
  anyOf?: SchemaNode[];
  $ref?: string;
  $defs?: Record<string, SchemaNode>;
  definitions?: Record<string, SchemaNode>;
}

/** One renderable property: its key plus its (ref-resolved) schema node. */
export interface SchemaProp {
  key: string;
  schema: SchemaNode;
}

/**
 * Collapse a nullable union (`anyOf: [{type:T}, {type:null}]`) down to its
 * concrete branch so the dynamic-form renderer's scalar `type` checks match.
 *
 * Pydantic V2 serializes `int | None` / `float | None` (e.g. the sampling
 * `num_frames` / `fps` fields) as an `anyOf` with NO scalar `type`, and hangs
 * the field-level extras (`min`/`max`/`step`/`description`/`group`/`default`/
 * `title`) on the PARENT node — not inside the branches. Without this, the
 * field component matched none of `isNumber()`/`isString()`/`isBoolean()` and
 * rendered the label + description with NO input control (the "sampling
 * settings not fully visible" bug). We hoist the concrete branch's `type`
 * (and any nested `items`/`enum`) while keeping every parent-level key.
 *
 * Nodes that already carry a scalar `type`, or whose `anyOf` is all-null /
 * multi-concrete (a genuine polymorphic union we don't render), are returned
 * unchanged.
 */
export function collapseNullableUnion(node: SchemaNode): SchemaNode {
  if (!node || node.type || !Array.isArray(node.anyOf)) return node;
  const concrete = node.anyOf.filter((b) => b && b.type && b.type !== 'null');
  if (concrete.length !== 1) return node;
  const { anyOf, ...rest } = node;
  // Branch supplies `type` (+ nested `items`/`enum`); parent-level keys win for
  // everything else. The trailing `type` is explicit so an undefined parent
  // `type` key can't shadow the branch's concrete type.
  return { ...concrete[0], ...rest, type: concrete[0].type };
}

/**
 * Coerce one form value to its schema-declared type. Numeric leaves
 * (`integer`/`number`): a numeric STRING becomes a number; `''` → `null` (so an
 * emptied optional clears rather than sending an invalid empty string);
 * non-numeric junk passes through untouched. Arrays and nested objects recurse.
 *
 * This exists because numeric inputs render as `[type]="'number'"` (a property
 * binding), so Angular's `NumberValueAccessor` — selected by the *static*
 * `input[type=number]` attribute — never attaches. The string
 * `DefaultValueAccessor` is used instead, so an EDITED numeric field holds a
 * string ("0", "25", "0.3") while an untouched one keeps its numeric default.
 * Both `int` `0` and string `"0"` looked the same in the UI, but the backend
 * stores the config verbatim and the trainer reads it raw — a stray `"0"`
 * silently zeroed video fps resolution (target_fps must be > 0). Coercing
 * against the schema at submit makes the persisted config always correctly typed.
 */
export function coerceSchemaValue(value: unknown, node: SchemaNode): unknown {
  const t = node.type;
  if (t === 'integer' || t === 'number') {
    if (typeof value !== 'string') return value;
    const s = value.trim();
    if (s === '') return null;
    const n = Number(s);
    if (!Number.isFinite(n)) return value;
    return t === 'integer' ? Math.trunc(n) : n;
  }
  if (t === 'array' && Array.isArray(value) && node.items) {
    const items = collapseNullableUnion(node.items);
    return value.map((v) => coerceSchemaValue(v, items));
  }
  if (node.properties && value && typeof value === 'object' && !Array.isArray(value)) {
    const obj = value as Record<string, unknown>;
    for (const [k, ps] of Object.entries(node.properties)) {
      if (k in obj) obj[k] = coerceSchemaValue(obj[k], collapseNullableUnion(ps));
    }
    return obj;
  }
  return value;
}

/**
 * Coerce stringified numeric fields in a training-config object to real
 * numbers, IN PLACE, against the root schema's declared properties. Recurses
 * into nested objects (the datasets array) and numeric arrays (resolutions).
 * Non-schema keys and already-typed values are left untouched.
 */
export function coerceConfigNumbers(
  config: Record<string, unknown>,
  schema: SchemaNode | undefined,
): void {
  const props = schema?.properties;
  if (!props || !config || typeof config !== 'object') return;
  for (const [key, rawProp] of Object.entries(props)) {
    if (!(key in config)) continue;
    config[key] = coerceSchemaValue(config[key], collapseNullableUnion(rawProp));
  }
}
