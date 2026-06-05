/** One image queued for batch crop. `targetWidth/Height` come from the
 *  analysis pass (the harmonized crop target for that image). */
export interface CropAllItem {
    path: string;
    targetWidth: number;
    targetHeight: number;
}
