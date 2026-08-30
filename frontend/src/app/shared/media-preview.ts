/**
 * Dataset preview-image URL helpers.
 *
 * A dataset cover is painted into a card roughly 260–630px wide, but a
 * `preview_image` is just the first multimedia file in the folder — a training
 * source at its full authored size. Measured on the library grid: 94 covers,
 * median 2.36 MP and one at 58 MP (9339x6223), 598 MP of decoded bitmap for
 * 3.9 MP of screen. That is what made fast scrolling impossible; with the
 * images hidden the identical scroll ran at a flat 60fps. So every cover routes
 * through `GET /datasets/{name}/thumbnail`, which serves a bounded WebP.
 *
 * Two exceptions, both deliberate:
 *  - Animated GIFs keep their direct `/media` URL, because a thumbnail is one
 *    still frame and a GIF cover is meant to animate.
 *  - A thumbnail that 404s falls back to the direct URL (see
 *    `directDatasetMediaUrl`). Thumbnail generation is Pillow-based, so a
 *    format the browser can paint but this install's Pillow cannot decode
 *    (AVIF without `pillow-avif-plugin`) must still show a cover rather than
 *    a hole.
 *
 * Video clips have always come through here: a browser cannot paint an
 * mp4/webm/mkv/avi in an `<img>` at all, so those get a first-frame poster.
 */

/** Video container formats a browser cannot render inside an `<img>`. */
const POSTER_VIDEO_EXTENSIONS = ['.mp4', '.webm', '.mkv', '.avi'];

/**
 * Rendition width requested for dataset covers.
 *
 * The library grid caps at 6 columns, so a card is `(width - 60px) / 6` — and
 * that is CSS pixels, which the display's pixel ratio then multiplies. 512 was
 * chosen against a 258px card at DPR 1 and shipped visibly soft covers on a
 * real monitor, so this is sized for the case that actually bites: a wide card
 * on a HiDPI display. At 1024 the whole 95-dataset library is ~55 MP of bitmap
 * against 598 MP for the originals, and the scroll still measures a flat 60fps.
 *
 * Must be a member of `ALLOWED_MAX_EDGES` in
 * `backend/app/core/dataset/thumbnails.py` — the endpoint rejects anything else.
 */
export const PREVIEW_MAX_EDGE = 1024;

/**
 * True when `filename` is a video clip that must be previewed via a poster
 * thumbnail rather than rendered directly in an `<img>`.
 */
export function needsVideoPoster(filename: string): boolean {
    const lower = filename.toLowerCase();
    return POSTER_VIDEO_EXTENSIONS.some(ext => lower.endsWith(ext));
}

/** True when `filename` is a GIF, which stays live rather than becoming a poster. */
export function staysAnimated(filename: string): boolean {
    return filename.toLowerCase().endsWith('.gif');
}

/**
 * Direct `/media` URL for a dataset file — the original bytes, unresized.
 *
 * This is the fallback when a thumbnail is unavailable, and the primary URL
 * for animated GIFs. Prefer `datasetPreviewUrl` for anything card-sized.
 */
export function directDatasetMediaUrl(
    mediaBaseUrl: string,
    datasetName: string,
    previewImage: string,
): string {
    // `previewImage` is a dataset-relative path and may contain sub-directories.
    // Encoding is transparent to those — the server decodes before routing — but
    // it is what keeps a `#` or `?` in a filename from truncating the URL.
    return `${mediaBaseUrl}/${encodeURIComponent(datasetName)}/${encodeURIComponent(previewImage)}`;
}

/**
 * Build the preview-image URL for a dataset cover/thumbnail.
 *
 * Everything but an animated GIF resolves to a bounded WebP rendition from the
 * thumbnail endpoint; see the module comment for why.
 */
export function datasetPreviewUrl(
    apiUrl: string,
    mediaBaseUrl: string,
    datasetName: string,
    previewImage: string,
): string {
    if (staysAnimated(previewImage)) {
        return directDatasetMediaUrl(mediaBaseUrl, datasetName, previewImage);
    }
    return (
        `${apiUrl}/datasets/${encodeURIComponent(datasetName)}/thumbnail` +
        `?image_rel_path=${encodeURIComponent(previewImage)}` +
        `&max_edge=${PREVIEW_MAX_EDGE}`
    );
}
