import { buildDisplayUrl, buildPixelSourceUrl, withCorsParam } from './edit-canvas.component';

describe('withCorsParam', () => {
    it('appends ?cors=1 to a bare URL', () => {
        expect(withCorsParam('http://h:8000/media/ds/a.jpg'))
            .toBe('http://h:8000/media/ds/a.jpg?cors=1');
    });

    it('appends &cors=1 when the URL already has a query', () => {
        expect(withCorsParam('http://h:8000/media/ds/a.jpg?r=2'))
            .toBe('http://h:8000/media/ds/a.jpg?r=2&cors=1');
    });

    it('appends &cors=1 to an overlay URL with a hash query', () => {
        expect(withCorsParam('http://h:8000/media/ds/ov.png?h=abc&r=1'))
            .toBe('http://h:8000/media/ds/ov.png?h=abc&r=1&cors=1');
    });
});

describe('buildDisplayUrl', () => {
    const BASE = 'http://h:8000/media';

    it('returns the original URL when no overlay is present', () => {
        expect(buildDisplayUrl(BASE, 'My DS', 'a.jpg', null, 0))
            .toBe('http://h:8000/media/My%20DS/a.jpg');
    });

    it('appends a cache-bust rev to the original when sourceRev > 0', () => {
        expect(buildDisplayUrl(BASE, 'ds', 'a.jpg', null, 3))
            .toBe('http://h:8000/media/ds/a.jpg?r=3');
    });

    it('returns the overlay PNG path with a hash query when an overlay exists', () => {
        const ov = { dataset_name: 'ds', overlay_file: 'overlays/a.png', hash: 'abc123' };
        expect(buildDisplayUrl(BASE, 'ds', 'a.jpg', ov, 0))
            .toBe('http://h:8000/media/ds/overlays/a.png?h=abc123');
    });

    it('falls back to a fixed "?h=ov" key when the overlay has no hash', () => {
        const ov = { dataset_name: 'ds', overlay_file: 'overlays/a.png' };
        expect(buildDisplayUrl(BASE, 'ds', 'a.jpg', ov, 0))
            .toBe('http://h:8000/media/ds/overlays/a.png?h=ov');
    });

    it('appends &r=<rev> to the overlay URL when sourceRev > 0', () => {
        const ov = { dataset_name: 'ds', overlay_file: 'overlays/a.png', hash: 'h1' };
        expect(buildDisplayUrl(BASE, 'ds', 'a.jpg', ov, 2))
            .toBe('http://h:8000/media/ds/overlays/a.png?h=h1&r=2');
    });
});

describe('buildPixelSourceUrl', () => {
    const BASE = 'http://h:8000/media';

    it('returns the ORIGINAL URL even when an overlay would be available', () => {
        // The canvas pixel pipeline must always source the original, because
        // the PreviewPipeline applies the recipe stored in sliders on top of
        // whatever it loads. Sourcing the overlay PNG (which already has the
        // recipe baked in) would double-apply on Save and on re-open.
        // User-reported: Save → reload → "the same overlay values are applied"
        // again, producing recipe-on-top-of-recipe.
        expect(buildPixelSourceUrl(BASE, 'My DS', 'a.jpg', 0))
            .toBe('http://h:8000/media/My%20DS/a.jpg');
    });

    it('cache-busts with ?r=<rev> after Bake/Revert', () => {
        expect(buildPixelSourceUrl(BASE, 'ds', 'a.jpg', 7))
            .toBe('http://h:8000/media/ds/a.jpg?r=7');
    });

    it('omits the rev query when sourceRev is 0', () => {
        expect(buildPixelSourceUrl(BASE, 'ds', 'a.jpg', 0))
            .toBe('http://h:8000/media/ds/a.jpg');
    });
});
