import { withCorsParam } from './edit-canvas.component';

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
