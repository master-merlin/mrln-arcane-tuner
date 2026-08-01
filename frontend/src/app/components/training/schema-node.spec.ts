import {
  coerceConfigNumbers,
  coerceSchemaValue,
  collapseNullableUnion,
  SchemaNode,
} from './schema-node';

/**
 * Regression: Pydantic V2 serializes `int | None` / `float | None` (the
 * sampling `num_frames` / `fps` fields) as `anyOf: [{type:T}, {type:null}]`
 * with NO scalar `type` on the node — so the dynamic-form-field's
 * `isNumber()`/`isString()` matched nothing and rendered the label +
 * description with NO input control. `collapseNullableUnion` hoists the
 * concrete branch's `type` so the renderer's scalar checks match again.
 */
describe('collapseNullableUnion()', () => {
  it('collapses Optional[int] (anyOf int|null) to a number schema, keeping field extras', () => {
    const node: SchemaNode = {
      anyOf: [{ type: 'integer' }, { type: 'null' }],
      default: null,
      title: 'Num Frames',
      description: 'Frames to sample for video models (None = still image)',
      min: 1,
      max: 256,
      step: 1,
      group: 'SAMPLING',
    };
    const out = collapseNullableUnion(node);
    expect(out.type).toBe('integer');
    expect(out.anyOf).toBeUndefined();
    // Parent-level extras survive the collapse (drive min/max/step + label).
    expect(out.min).toBe(1);
    expect(out.max).toBe(256);
    expect(out.step).toBe(1);
    expect(out.title).toBe('Num Frames');
    expect(out.group).toBe('SAMPLING');
  });

  it('collapses Optional[float] (anyOf number|null) to a number schema', () => {
    const out = collapseNullableUnion({ anyOf: [{ type: 'number' }, { type: 'null' }], step: 1.0 });
    expect(out.type).toBe('number');
    expect(out.step).toBe(1.0);
  });

  it('hoists a nested branch `items` (e.g. Optional[list[str]])', () => {
    const out = collapseNullableUnion({
      anyOf: [{ type: 'array', items: { type: 'string' } }, { type: 'null' }],
    });
    expect(out.type).toBe('array');
    expect(out.items).toEqual({ type: 'string' });
  });

  it('leaves a node that already has a scalar type untouched', () => {
    const node: SchemaNode = { type: 'integer', default: 1024 };
    expect(collapseNullableUnion(node)).toBe(node);
  });

  it('leaves a node without anyOf untouched', () => {
    const node: SchemaNode = { type: 'array', items: { type: 'string' } };
    expect(collapseNullableUnion(node)).toBe(node);
  });

  it('does NOT collapse a genuine multi-concrete union (we cannot render it as one input)', () => {
    const node: SchemaNode = { anyOf: [{ type: 'integer' }, { type: 'string' }] };
    const out = collapseNullableUnion(node);
    expect(out).toBe(node);
    expect(out.type).toBeUndefined();
  });

  it('does NOT collapse an all-null anyOf', () => {
    const node: SchemaNode = { anyOf: [{ type: 'null' }] };
    expect(collapseNullableUnion(node).type).toBeUndefined();
  });
});

/**
 * Regression: numeric inputs render as `[type]="'number'"` (a property
 * binding), so Angular's NumberValueAccessor — selected by the STATIC
 * `input[type=number]` — never attaches and the string DefaultValueAccessor is
 * used. An edited numeric field therefore submits a STRING ("0", "25", "0.3");
 * the backend stores the config verbatim and the trainer reads it raw, so a
 * stray target_fps "0" zeroed video fps resolution (G2/G4/G1-stills pre-cache
 * crash). coerceConfigNumbers types the payload at submit.
 */
describe('coerceSchemaValue()', () => {
  it('coerces a numeric string to a number (the target_fps="0" bug)', () => {
    expect(coerceSchemaValue('0', { type: 'integer' })).toBe(0);
    expect(coerceSchemaValue('25', { type: 'integer' })).toBe(25);
    expect(coerceSchemaValue('0.3', { type: 'number' })).toBe(0.3);
  });

  it('truncates to int for integer fields, keeps fractional for number fields', () => {
    expect(coerceSchemaValue('2.9', { type: 'integer' })).toBe(2);
    expect(coerceSchemaValue('2.9', { type: 'number' })).toBe(2.9);
  });

  it('leaves already-typed numbers and non-numeric junk untouched', () => {
    expect(coerceSchemaValue(24, { type: 'number' })).toBe(24);
    expect(coerceSchemaValue('abc', { type: 'integer' })).toBe('abc');
  });

  it('maps an empty numeric string to null (emptied optional clears)', () => {
    expect(coerceSchemaValue('', { type: 'integer' })).toBeNull();
    expect(coerceSchemaValue('   ', { type: 'number' })).toBeNull();
  });

  it('coerces each element of a numeric array (resolutions)', () => {
    expect(
      coerceSchemaValue(['768', '1024'], { type: 'array', items: { type: 'integer' } }),
    ).toEqual([768, 1024]);
  });

  it('recurses into nested object properties (a dataset row)', () => {
    const node: SchemaNode = {
      type: 'object',
      properties: { num_repeats: { type: 'integer' }, name: { type: 'string' } },
    };
    expect(coerceSchemaValue({ num_repeats: '3', name: 'set' }, node)).toEqual({
      num_repeats: 3,
      name: 'set',
    });
  });

  it('collapses a nullable numeric union before coercing (Optional[int])', () => {
    const node: SchemaNode = { anyOf: [{ type: 'integer' }, { type: 'null' }] };
    // coerceConfigNumbers collapses per-property; coerceSchemaValue is given the
    // collapsed node, so verify the collapse path end-to-end via the wrapper.
    const cfg: Record<string, unknown> = { num_frames: '25' };
    coerceConfigNumbers(cfg, { properties: { num_frames: node } } as SchemaNode);
    expect(cfg['num_frames']).toBe(25);
  });
});

describe('coerceConfigNumbers()', () => {
  const schema: SchemaNode = {
    type: 'object',
    properties: {
      target_fps: { type: 'number' },
      frame_stride: { type: 'integer' },
      temporal_coverage: { type: 'string', enum: ['first', 'tiled', 'sliding'] },
      cache_latents: { type: 'boolean' },
      resolutions: { type: 'array', items: { type: 'integer' } },
      datasets: {
        type: 'array',
        items: {
          type: 'object',
          properties: { num_repeats: { type: 'integer' }, dataset_name: { type: 'string' } },
        },
      },
    },
  };

  it('coerces the exact failing config shape (strings → numbers)', () => {
    const cfg: Record<string, unknown> = {
      target_fps: '0',
      frame_stride: '2',
      temporal_coverage: 'tiled',
      cache_latents: true,
      resolutions: [768],
      datasets: [{ num_repeats: 1, dataset_name: 'ds-a' }],
      project_id: 'p1', // non-schema key — must survive untouched
    };
    coerceConfigNumbers(cfg, schema);
    expect(cfg['target_fps']).toBe(0);
    expect(cfg['frame_stride']).toBe(2);
    expect(cfg['temporal_coverage']).toBe('tiled'); // enum string left alone
    expect(cfg['cache_latents']).toBe(true); // boolean left alone
    expect(cfg['resolutions']).toEqual([768]);
    expect(cfg['datasets']).toEqual([{ num_repeats: 1, dataset_name: 'ds-a' }]);
    expect(cfg['project_id']).toBe('p1');
  });

  it('coerces a stringified nested dataset numeric', () => {
    const cfg: Record<string, unknown> = {
      datasets: [{ num_repeats: '3', dataset_name: 'set' }],
    };
    coerceConfigNumbers(cfg, schema);
    expect(cfg['datasets']).toEqual([{ num_repeats: 3, dataset_name: 'set' }]);
  });

  it('is a no-op without a schema', () => {
    const cfg: Record<string, unknown> = { target_fps: '0' };
    coerceConfigNumbers(cfg, undefined);
    expect(cfg['target_fps']).toBe('0');
  });
});
