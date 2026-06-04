import { buildEditablePayload } from './template-json.component';
import { Template } from '../../services/template.service';

function tpl(over: Partial<Template>): Template {
    return {
        id: 'id', name: 'n', project_id: 'p', config: {},
        created_at: 0, updated_at: 0, used_count: 0,
        is_default: false, readonly: false,
        ...over,
    } as Template;
}

describe('buildEditablePayload', () => {
    it('includes name + config, excludes server-managed fields', () => {
        const out = buildEditablePayload(tpl({
            id: 'x', name: 'Mine', project_id: 'proj', config: { a: 1 },
            created_at: 123, updated_at: 456, used_count: 9,
            is_default: true, readonly: true, branched_from: 'b',
        }));
        expect(out).toEqual({ name: 'Mine', config: { a: 1 } });
        expect(out['id']).toBeUndefined();
        expect(out['project_id']).toBeUndefined();
        expect(out['readonly']).toBeUndefined();
    });

    it('includes system_prompt + model_id for captioning templates', () => {
        const out = buildEditablePayload(tpl({
            name: 'Cap', config: { temperature: 0.7 },
            system_prompt: 'Describe.', model_id: 'qwen3-vl',
        }));
        expect(out).toEqual({
            name: 'Cap', config: { temperature: 0.7 },
            system_prompt: 'Describe.', model_id: 'qwen3-vl',
        });
    });

    it('includes definition_id for training templates', () => {
        const out = buildEditablePayload(tpl({
            name: 'Train', config: { max_train_steps: 1000 }, definition_id: 'flux2-lora',
        }));
        expect(out['definition_id']).toBe('flux2-lora');
        expect(out['system_prompt']).toBeUndefined();
    });

    it('defaults a missing config to an empty object', () => {
        const out = buildEditablePayload(tpl({ name: 'NoCfg', config: undefined as any }));
        expect(out['config']).toEqual({});
    });
});
