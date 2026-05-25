/**
 * Pure geometry helpers for the LossChart component.
 *
 * Kept Angular-free so the math can be exercised in isolation under TDD.
 */

export function linearTicks(min: number, max: number, count: number): number[] {
    if (count < 2) return [min];
    const step = (max - min) / (count - 1);
    return Array.from({ length: count }, (_, i) => min + step * i);
}

export function mapY(value: number, min: number, max: number, height: number): number {
    if (max === min) return height / 2;
    return height - ((value - min) / (max - min)) * height;
}

export function ema(values: ReadonlyArray<number>, alpha: number): number[] {
    if (values.length === 0) return [];
    const out: number[] = [values[0]];
    for (let i = 1; i < values.length; i++) {
        out.push(alpha * values[i] + (1 - alpha) * out[i - 1]);
    }
    return out;
}
