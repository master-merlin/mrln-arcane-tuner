import { collapseNullableUnion, SchemaNode } from './schema-node';

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
