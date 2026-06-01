import { runCropAll, CropAllItem } from './crop-all';

function items(n: number): CropAllItem[] {
    return Array.from({ length: n }, (_, i) => ({ path: `img_${i}.jpg`, targetWidth: 512, targetHeight: 512 }));
}

describe('runCropAll', () => {
    it('crops every item sequentially and tallies successes', async () => {
        const seen: string[] = [];
        const progress: number[] = [];
        const res = await runCropAll(items(3), {
            origin: 'center',
            crop: async (it) => { seen.push(it.path); },
            onProgress: (p) => progress.push(p.current),
            isCancelled: () => false,
        });
        expect(seen).toEqual(['img_0.jpg', 'img_1.jpg', 'img_2.jpg']);
        expect(res.ok).toBe(3);
        expect(res.failed).toBe(0);
        expect(res.cancelled).toBe(false);
        expect(progress).toEqual([1, 2, 3]);
    });

    it('counts failures but keeps going', async () => {
        const res = await runCropAll(items(3), {
            origin: 'top',
            crop: async (it) => { if (it.path === 'img_1.jpg') throw new Error('boom'); },
            onProgress: () => {},
            isCancelled: () => false,
        });
        expect(res.ok).toBe(2);
        expect(res.failed).toBe(1);
    });

    it('stops early when cancelled', async () => {
        let n = 0;
        const res = await runCropAll(items(5), {
            origin: 'center',
            crop: async () => { n++; },
            onProgress: () => {},
            isCancelled: () => n >= 2,
        });
        expect(res.cancelled).toBe(true);
        expect(n).toBe(2);
        expect(res.ok).toBe(2);
    });

    it('passes the chosen origin through to each crop call', async () => {
        const origins: string[] = [];
        await runCropAll(items(2), {
            origin: 'bottom-right',
            crop: async (_it, origin) => { origins.push(origin); },
            onProgress: () => {},
            isCancelled: () => false,
        });
        expect(origins).toEqual(['bottom-right', 'bottom-right']);
    });
});
