import { TestBed } from '@angular/core/testing';
import { GridSkeletonComponent } from '../grid-skeleton.component';

/**
 * LANE-58 — the browse grid's placeholder while the first `/pairs` is in
 * flight. Asserts on the rendered DOM: the boxes exist, each carries the
 * loader dots, and the count follows `slots`.
 */
describe('GridSkeletonComponent', () => {
    function mount(slots: number, columns = 5) {
        TestBed.configureTestingModule({ imports: [GridSkeletonComponent] });
        const fixture = TestBed.createComponent(GridSkeletonComponent);
        fixture.componentRef.setInput('slots', slots);
        fixture.componentRef.setInput('columns', columns);
        fixture.detectChanges();
        return fixture;
    }

    it('draws one placeholder tile per slot, each with loader dots', () => {
        const fixture = mount(4);
        const host: HTMLElement = fixture.nativeElement;
        const tiles = host.querySelectorAll('[data-testid="grid-skeleton-tile"]');
        expect(tiles.length).toBe(4);
        for (const t of Array.from(tiles)) {
            expect(t.querySelectorAll('.grid-thumb-loader > span').length).toBe(3);
        }
        const grid = host.querySelector<HTMLElement>('[data-testid="grid-skeleton"]')!;
        expect(grid.getAttribute('aria-busy')).toBe('true');
        expect(grid.style.gridTemplateColumns).toBe('repeat(5, minmax(0, 1fr))');
    });

    it('follows the requested column count so the swap to real tiles does not reflow', () => {
        const fixture = mount(2, 3);
        const grid = fixture.nativeElement.querySelector('[data-testid="grid-skeleton"]') as HTMLElement;
        expect(grid.style.gridTemplateColumns).toBe('repeat(3, minmax(0, 1fr))');
    });

    it('draws nothing for zero, negative or non-finite slots', () => {
        for (const n of [0, -3, Number.NaN, Number.POSITIVE_INFINITY]) {
            TestBed.resetTestingModule();
            const fixture = mount(n);
            expect(fixture.nativeElement.querySelectorAll('[data-testid="grid-skeleton-tile"]').length).toBe(0);
        }
    });
});
