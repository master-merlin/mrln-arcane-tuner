import { TestBed } from '@angular/core/testing';
import { Subject } from 'rxjs';
import { NavigationEnd, Router } from '@angular/router';
import {
    ALL_DATASET_SEARCH_FIELDS,
    SearchStore,
    type DatasetSearchField,
} from '../search.store';

class RouterStub {
    events = new Subject<unknown>();
}

describe('SearchStore', () => {
    let store: SearchStore;
    let router: RouterStub;

    beforeEach(() => {
        router = new RouterStub();
        TestBed.configureTestingModule({
            providers: [
                SearchStore,
                { provide: Router, useValue: router },
            ],
        });
        store = TestBed.inject(SearchStore);
    });

    it('defaults to empty query and all fields enabled', () => {
        expect(store.query()).toBe('');
        expect(store.fields().size).toBe(ALL_DATASET_SEARCH_FIELDS.length);
        for (const f of ALL_DATASET_SEARCH_FIELDS) {
            expect(store.fields().has(f)).toBeTrue();
        }
    });

    it('setField removes a field when disabled', () => {
        store.setField('tags', false);
        expect(store.fields().has('tags')).toBeFalse();
        expect(store.fields().size).toBe(ALL_DATASET_SEARCH_FIELDS.length - 1);
    });

    it('setField adds a field back when enabled', () => {
        store.setField('tags', false);
        store.setField('tags', true);
        expect(store.fields().has('tags')).toBeTrue();
    });

    it('resetFields restores the full default set', () => {
        store.setField('tags', false);
        store.setField('notes', false);
        store.resetFields();
        expect(store.fields().size).toBe(ALL_DATASET_SEARCH_FIELDS.length);
    });

    it('clears query when navigating away from /datasets', () => {
        store.query.set('vehicle');
        router.events.next(makeNavEnd('/projects'));
        expect(store.query()).toBe('');
    });

    it('preserves query while navigating within /datasets', () => {
        store.query.set('vehicle');
        router.events.next(makeNavEnd('/datasets'));
        expect(store.query()).toBe('vehicle');
    });

    it('preserves field selection across route changes', () => {
        store.setField('tags', false);
        router.events.next(makeNavEnd('/projects'));
        expect(store.fields().has('tags')).toBeFalse();
    });
});

function makeNavEnd(url: string): NavigationEnd {
    return new NavigationEnd(1, url, url);
}
