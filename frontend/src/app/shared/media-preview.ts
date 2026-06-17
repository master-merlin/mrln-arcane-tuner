/**
 * Dataset preview-image URL helpers.
 *
 * A dataset's `preview_image` is just the first multimedia file in the folder,
 * which for a video-only dataset is a video clip (e.g. `clip001.mp4`). Rendering
 * that filename in an `<img>` fails — browsers can't paint an mp4/webm/mkv/avi
 * in an image element — so the library card and the training dataset picker
 * showed a broken/placeholder image. These helpers route such clips through the
 * thumbnail endpoint instead, which serves a first-frame WebP poster (the same
 * poster the dataset grid uses for its video tiles).
 *
 * Animated GIFs are intentionally NOT treated as posters: they animate fine in
 * an `<img>`, so they keep their direct `/media` URL and stay live.
 */

/** Video container formats a browser cannot render inside an `<img>`. */
const POSTER_VIDEO_EXTENSIONS = ['.mp4', '.webm', '.mkv', '.avi'];

/**
 * True when `filename` is a video clip that must be previewed via a poster
 * thumbnail rather than rendered directly in an `<img>`.
 */
export function needsVideoPoster(filename: string): boolean {
    const lower = filename.toLowerCase();
    return POSTER_VIDEO_EXTENSIONS.some(ext => lower.endsWith(ext));
}

/**
 * Build the preview-image URL for a dataset cover/thumbnail.
 *
 * Stills and animated GIFs render directly from `/media` (byte-identical to the
 * legacy behaviour). Video clips route through `GET /datasets/{name}/thumbnail`,
 * which extracts and serves a 256px first-frame WebP poster.
 */
export function datasetPreviewUrl(
    apiUrl: string,
    mediaBaseUrl: string,
    datasetName: string,
    previewImage: string,
): string {
    const name = encodeURIComponent(datasetName);
    if (needsVideoPoster(previewImage)) {
        return (
            `${apiUrl}/datasets/${name}/thumbnail` +
            `?image_rel_path=${encodeURIComponent(previewImage)}`
        );
    }
    return `${mediaBaseUrl}/${name}/${previewImage}`;
}
