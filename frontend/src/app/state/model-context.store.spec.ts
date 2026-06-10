// frontend/src/app/state/model-context.store.spec.ts
import { ModelContextStore, type DefinitionRef } from './model-context.store';

const DEF: DefinitionRef = { id: 'flux1-schnell', family: 'flux1', name: 'Flux.1 Schnell' };

describe('ModelContextStore', () => {
    beforeEach(() => localStorage.clear());

    it('defaults to model-aware off and no active definition', () => {
        const store = new ModelContextStore();
        expect(store.modelAware()).toBe(false);
        expect(store.activeDefinition()).toBeNull();
    });

    it('setModelAware(false) clears the active definition', () => {
        const store = new ModelContextStore();
        store.setModelAware(true);
        store.setDefinition(DEF);
        store.setModelAware(false);
        expect(store.modelAware()).toBe(false);
        expect(store.activeDefinition()).toBeNull();
    });

    it('persists model-aware + definition across instances', () => {
        const a = new ModelContextStore();
        a.setModelAware(true);
        a.setDefinition(DEF);

        const b = new ModelContextStore();
        expect(b.modelAware()).toBe(true);
        expect(b.activeDefinition()).toEqual(DEF);
    });

    it('activeDefinitionId is null when model-aware is off even if a def was set', () => {
        const store = new ModelContextStore();
        store.setModelAware(true);
        store.setDefinition(DEF);
        store.setModelAware(false);
        expect(store.activeDefinitionId()).toBeNull();
    });
});
