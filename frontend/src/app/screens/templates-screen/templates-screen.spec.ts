import { describe, it, expect } from 'vitest';
import { TemplatesScreen } from './templates-screen';

const T = (over: Partial<{ id: string; name: string; project_id: string|null; is_default: boolean; readonly: boolean; definition_id: string; model_id: string }>) => ({
  id: 'x', name: 'n', project_id: null, is_default: false, readonly: false, config: {}, created_at: 0, updated_at: 0, used_count: 0, ...over,
});
const R = (domain: 'training'|'captioning'|'masking', scopeId: string|null, over = {}) =>
  ({ domain, scopeId, scopeLabel: scopeId ?? 'General', tpl: T({ ...over }) as never });

describe('TemplatesScreen.filterRows', () => {
  const rows = [
    R('training', null, { name: 'Anime', definition_id: 'flux-dev' }),
    R('captioning', 'p1', { name: 'Caps', model_id: 'qwen3-vl', is_default: true }),
    R('masking', 'p1', { name: 'Mask', model_id: 'sam3', readonly: true }),
  ];
  it('filters by domain', () => {
    expect(TemplatesScreen.filterRows(rows, { domain: 'training', scope: 'all', search: '', flag: 'all' })).toHaveLength(1);
  });
  it('filters by scope general vs project', () => {
    expect(TemplatesScreen.filterRows(rows, { domain: 'all', scope: 'general', search: '', flag: 'all' })).toHaveLength(1);
    expect(TemplatesScreen.filterRows(rows, { domain: 'all', scope: 'p1', search: '', flag: 'all' })).toHaveLength(2);
  });
  it('search matches name, definition_id and model_id', () => {
    expect(TemplatesScreen.filterRows(rows, { domain: 'all', scope: 'all', search: 'anime', flag: 'all' })).toHaveLength(1); // name
    expect(TemplatesScreen.filterRows(rows, { domain: 'all', scope: 'all', search: 'flux', flag: 'all' })).toHaveLength(1); // definition_id
    expect(TemplatesScreen.filterRows(rows, { domain: 'all', scope: 'all', search: 'qwen', flag: 'all' })).toHaveLength(1); // model_id
  });
  it('filters by default/system flag', () => {
    expect(TemplatesScreen.filterRows(rows, { domain: 'all', scope: 'all', search: '', flag: 'default' })).toHaveLength(1);
    expect(TemplatesScreen.filterRows(rows, { domain: 'all', scope: 'all', search: '', flag: 'system' })).toHaveLength(1);
  });
});
