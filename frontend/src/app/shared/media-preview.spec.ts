import { datasetPreviewUrl, needsVideoPoster } from './media-preview';

describe('needsVideoPoster', () => {
    it('flags non-image video containers (case-insensitive)', () => {
        expect(needsVideoPoster('clip001.mp4')).toBe(true);
        expect(needsVideoPoster('shot.webm')).toBe(true);
        expect(needsVideoPoster('take.MKV')).toBe(true);
        expect(needsVideoPoster('legacy.AVI')).toBe(true);
    });

    it('does NOT flag stills or animated GIFs (they render in <img>)', () => {
        expect(needsVideoPoster('photo.jpg')).toBe(false);
        expect(needsVideoPoster('art.png')).toBe(false);
        expect(needsVideoPoster('scan.webp')).toBe(false);
        expect(needsVideoPoster('loop.gif')).toBe(false);
    });
});

describe('datasetPreviewUrl', () => {
    const api = '/api';
    const media = '/media';

    it('routes video clips through the thumbnail (poster) endpoint', () => {
        expect(datasetPreviewUrl(api, media, 'My Set', 'clip001.mp4')).toBe(
            '/api/datasets/My%20Set/thumbnail?image_rel_path=clip001.mp4',
        );
    });

    it('serves stills directly from /media (unchanged legacy URL)', () => {
        // Dataset name is encoded; the filename is left as-is to match the
        // pre-existing previewUrl behaviour exactly.
        expect(datasetPreviewUrl(api, media, 'My Set', 'cat.jpg')).toBe(
            '/media/My%20Set/cat.jpg',
        );
    });

    it('keeps animated GIFs on the direct /media URL (stays live)', () => {
        expect(datasetPreviewUrl(api, media, 'anims', 'loop.gif')).toBe(
            '/media/anims/loop.gif',
        );
    });
});
