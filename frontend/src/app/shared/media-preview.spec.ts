import {
    PREVIEW_MAX_EDGE,
    datasetPreviewUrl,
    directDatasetMediaUrl,
    needsVideoPoster,
    staysAnimated,
} from './media-preview';

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

describe('staysAnimated', () => {
    it('flags GIFs only, case-insensitively', () => {
        expect(staysAnimated('loop.gif')).toBe(true);
        expect(staysAnimated('LOOP.GIF')).toBe(true);
        expect(staysAnimated('cat.jpg')).toBe(false);
        expect(staysAnimated('clip.mp4')).toBe(false);
    });
});

describe('datasetPreviewUrl', () => {
    const api = '/api';
    const media = '/media';

    it('routes video clips through the thumbnail (poster) endpoint', () => {
        expect(datasetPreviewUrl(api, media, 'My Set', 'clip001.mp4')).toBe(
            `/api/datasets/My%20Set/thumbnail?image_rel_path=clip001.mp4&max_edge=${PREVIEW_MAX_EDGE}`,
        );
    });

    it('routes stills through the thumbnail endpoint too, at a bounded edge', () => {
        // Deliberate contract change. Stills used to serve the original bytes
        // from /media, which meant a card 260px wide decoded whatever the
        // training source happened to be — up to 58 MP on the measured library.
        // That is the whole cause of the scroll stutter, so a still is a
        // rendition now, exactly like a video poster.
        expect(datasetPreviewUrl(api, media, 'My Set', 'cat.jpg')).toBe(
            `/api/datasets/My%20Set/thumbnail?image_rel_path=cat.jpg&max_edge=${PREVIEW_MAX_EDGE}`,
        );
    });

    it('requests an edge large enough for a wide card on a HiDPI display', () => {
        // Regression: 512 was picked against a 258px card at DPR 1 and shipped
        // visibly soft covers on a real monitor. Card width is CSS pixels; the
        // pixel ratio multiplies it. Do not lower this without measuring on a
        // scaled display.
        expect(PREVIEW_MAX_EDGE).toBeGreaterThanOrEqual(1024);
    });

    it('keeps animated GIFs on the direct /media URL (stays live)', () => {
        // A thumbnail is one still frame; a GIF cover is meant to animate.
        expect(datasetPreviewUrl(api, media, 'anims', 'loop.gif')).toBe(
            '/media/anims/loop.gif',
        );
    });

    it('encodes sub-directory paths and shell-hostile filenames', () => {
        expect(datasetPreviewUrl(api, media, 'ds', 'sub/shot#3.png')).toBe(
            `/api/datasets/ds/thumbnail?image_rel_path=sub%2Fshot%233.png&max_edge=${PREVIEW_MAX_EDGE}`,
        );
    });
});

describe('directDatasetMediaUrl', () => {
    it('serves the original bytes, encoding both name and path', () => {
        expect(directDatasetMediaUrl('/media', 'My Set', 'sub/cat#1.jpg')).toBe(
            '/media/My%20Set/sub%2Fcat%231.jpg',
        );
    });
});
