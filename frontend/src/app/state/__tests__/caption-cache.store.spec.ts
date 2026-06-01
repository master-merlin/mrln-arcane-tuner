import { TestBed } from '@angular/core/testing';
import { CaptionCacheStore } from '../caption-cache.store';

describe('CaptionCacheStore', () => {
    let store: CaptionCacheStore;

    beforeEach(() => {
        TestBed.configureTestingModule({ providers: [CaptionCacheStore] });
        store = TestBed.inject(CaptionCacheStore);
    });

    it('seed then get returns the rows', () => {
        store.seed('ds1', new Map([['a.png', { caption_content: 'hi' }]]));
        expect(store.get('ds1').get('a.png')?.caption_content).toBe('hi');
    });

    it('get returns an empty map for an unknown dataset', () => {
        expect(store.get('nope').size).toBe(0);
    });

    it('setCaption(original) preserves an existing masked caption', () => {
        store.setRow('ds1', 'a.png', { masked_caption_content: 'm' });
        store.setCaption('ds1', 'a.png', 'orig', false);
        const row = store.get('ds1').get('a.png');
        expect(row?.caption_content).toBe('orig');
        expect(row?.masked_caption_content).toBe('m');
    });

    it('setCaption(masked) preserves an existing original caption', () => {
        store.setRow('ds1', 'a.png', { caption_content: 'o' });
        store.setCaption('ds1', 'a.png', 'masked', true);
        const row = store.get('ds1').get('a.png');
        expect(row?.masked_caption_content).toBe('masked');
        expect(row?.caption_content).toBe('o');
    });

    it('byDataset hands out a fresh map reference on each write (reactive)', () => {
        store.setCaption('ds1', 'a.png', 'x', false);
        const before = store.byDataset()['ds1'];
        store.setCaption('ds1', 'b.png', 'y', false);
        expect(store.byDataset()['ds1']).not.toBe(before);
    });

    it('remove drops a single row; clear drops the dataset', () => {
        store.seed('ds1', new Map([
            ['a.png', { caption_content: 'x' }],
            ['b.png', { caption_content: 'y' }],
        ]));
        store.remove('ds1', 'a.png');
        expect(store.get('ds1').has('a.png')).toBe(false);
        expect(store.get('ds1').has('b.png')).toBe(true);
        store.clear('ds1');
        expect(store.byDataset()['ds1']).toBeUndefined();
    });

    it('seed replaces all rows for an existing dataset', () => {
        store.seed('ds1', new Map([['a.png', { caption_content: 'old' }]]));
        store.seed('ds1', new Map([['b.png', { caption_content: 'new' }]]));
        expect(store.get('ds1').has('a.png')).toBe(false);
        expect(store.get('ds1').get('b.png')?.caption_content).toBe('new');
    });
});
