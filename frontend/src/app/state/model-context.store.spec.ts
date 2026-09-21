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

    it('setModelAware(false) deactivates the definition (activeDefinition null)', () => {
        const store = new ModelContextStore();
        store.setModelAware(true);
        store.setDefinition(DEF);
        store.setModelAware(false);
        expect(store.modelAware()).toBe(false);
        expect(store.activeDefinition()).toBeNull();
    });

    it('retains the definition across an off/on toggle for quick comparison', () => {
        const store = new ModelContextStore();
        store.setModelAware(true);
        store.setDefinition(DEF);
        store.setModelAware(false);
        expect(store.activeDefinition()).toBeNull();   // hidden while off
        store.setModelAware(true);
        expect(store.activeDefinition()).toEqual(DEF);  // restored — no re-pick
    });

    it('persists model-aware + definition across instances', () => {
        const a = new ModelContextStore();
        a.setModelAware(true);
        a.setDefinition(DEF);

        const b = new ModelContextStore();
        expect(b.modelAware()).toBe(true);
        expect(b.activeDefinition()).toEqual(DEF);
    });

    it('activeFrameRule is the active definition\'s served frame_rule, null when off or absent', () => {
        const store = new ModelContextStore();
        store.setModelAware(true);
        store.setDefinition({ ...DEF, frame_rule: '19n+5' });
        expect(store.activeFrameRule()).toBe('19n+5');
        // Survives a reload (the persisted definition carries it).
        expect(new ModelContextStore().activeFrameRule()).toBe('19n+5');
        store.setModelAware(false);
        expect(store.activeFrameRule()).toBeNull(); // model-agnostic browsing
        store.setModelAware(true);
        store.setDefinition({ ...DEF, frame_rule: null }); // an image definition
        expect(store.activeFrameRule()).toBeNull();
        store.setDefinition(DEF); // persisted before the backend served the field
        expect(store.activeFrameRule()).toBeNull();
    });

    it('activeIngestFps is the active definition\'s served ingest_fps, null when off, absent or not positive', () => {
        const store = new ModelContextStore();
        store.setModelAware(true);
        store.setDefinition({ ...DEF, ingest_fps: 24.0 });
        expect(store.activeIngestFps()).toBe(24);
        expect(new ModelContextStore().activeIngestFps()).toBe(24); // survives a reload
        store.setModelAware(false);
        expect(store.activeIngestFps()).toBeNull();
        store.setModelAware(true);
        store.setDefinition({ ...DEF, ingest_fps: null }); // every family but the fixed-clock one
        expect(store.activeIngestFps()).toBeNull();
        store.setDefinition({ ...DEF, ingest_fps: 0 });
        expect(store.activeIngestFps()).toBeNull();
        store.setDefinition(DEF); // persisted before the backend served the field
        expect(store.activeIngestFps()).toBeNull();
    });

    it('activeDefinitionId is null when model-aware is off even if a def was set', () => {
        const store = new ModelContextStore();
        store.setModelAware(true);
        store.setDefinition(DEF);
        store.setModelAware(false);
        expect(store.activeDefinitionId()).toBeNull();
    });
});
