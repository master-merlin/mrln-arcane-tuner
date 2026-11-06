import { signal } from '@angular/core';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { GPUStatus, SystemService, SystemSnapshot } from '../../../services/system.service';
import { JobStore } from '../../../state/job.store';
import { SystemMonitorComponent } from './system-monitor';

const CARD: GPUStatus = {
    index: 0,
    name: 'NVIDIA GeForce RTX 5090',
    vram_used_mb: 4096,
    vram_total_mb: 32768,
    vram_free_mb: 28672,
    vram_percent: 12.5,
    temperature_c: 71,
    power_draw_w: 412.5,
    power_limit_w: 600,
    gpu_utilization: 98,
    memory_utilization: 40,
    clock_graphics_mhz: 2610,
    clock_memory_mhz: 13801,
    processes: [],
};

describe('SystemMonitorComponent: the card\'s heat, fan and throttling', () => {
    let fixture: ComponentFixture<SystemMonitorComponent>;
    const metrics = signal<SystemSnapshot | null>(null);

    function show(fields: Partial<GPUStatus>): void {
        metrics.set({
            gpus: [{ ...CARD, ...fields }],
            system: { ram_used_mb: 1024, ram_total_mb: 2048, ram_percent: 50, cpu_percent: 10 },
        });
        TestBed.tick();
    }

    function byTestId(id: string): HTMLElement | null {
        return (fixture.nativeElement as HTMLElement).querySelector(`[data-testid="${id}"]`);
    }

    beforeEach(() => {
        metrics.set(null);
        TestBed.configureTestingModule({
            imports: [SystemMonitorComponent],
            providers: [
                {
                    provide: SystemService,
                    useValue: {
                        metrics,
                        subscribeMetrics: vi.fn(),
                        unsubscribeMetrics: vi.fn(),
                    },
                },
                {
                    provide: JobStore,
                    useValue: { entities: signal([]), loadHistory: vi.fn() },
                },
            ],
        });
        fixture = TestBed.createComponent(SystemMonitorComponent);
        fixture.detectChanges();
    });

    it('shows the margin and the fan with their values', () => {
        show({ temperature_margin_c: 12, fan_percent: 64 });
        expect(byTestId('sys-temp-margin')?.textContent).toContain('12°C');
        expect(byTestId('sys-fan')?.textContent).toContain('64%');
    });

    it('shows a real zero, as a stopped fan and a card at its limit are readings', () => {
        show({ temperature_margin_c: 0, fan_percent: 0 });
        expect(byTestId('sys-temp-margin')?.textContent).toContain('0°C');
        expect(byTestId('sys-fan')?.textContent).toContain('0%');
    });

    it('hides the margin when it is null and keeps the fan', () => {
        show({ temperature_margin_c: null, fan_percent: 64 });
        expect(byTestId('sys-temp-margin')).toBeNull();
        expect(byTestId('sys-fan')?.textContent).toContain('64%');
    });

    it('hides the fan when it is null and keeps the margin', () => {
        show({ temperature_margin_c: 12, fan_percent: null });
        expect(byTestId('sys-fan')).toBeNull();
        expect(byTestId('sys-temp-margin')?.textContent).toContain('12°C');
    });

    it('shows no empty tile when the card offers neither, or a backend sends neither', () => {
        show({ temperature_margin_c: null, fan_percent: null, throttle_reasons: null });
        expect(byTestId('sys-gpu-health')).toBeNull();
        expect(byTestId('sys-throttle-chip')).toBeNull();

        show({});
        expect(byTestId('sys-gpu-health')).toBeNull();
        expect(byTestId('sys-throttle-chip')).toBeNull();
        expect(byTestId('sys-unreadable-chip')).toBeNull();
    });

    for (const reason of ['sw_thermal', 'hw_thermal', 'hw_slowdown', 'hw_power_brake']) {
        it(`shows the throttle chip for ${reason}, naming it in the tooltip`, () => {
            show({ throttle_reasons: [reason] });
            const chip = byTestId('sys-throttle-chip');
            expect(chip?.textContent).toContain('Throttling');
            expect(chip?.title).toContain(reason);
        });
    }

    for (const reason of ['sw_power_cap', 'idle', 'display_clock', 'unknown_0x400']) {
        it(`shows no throttle chip for ${reason}`, () => {
            show({ throttle_reasons: [reason] });
            expect(byTestId('sys-throttle-chip')).toBeNull();
        });
    }

    it('shows the chip when a slowdown reason comes with a routine one, naming only the slowdown', () => {
        show({ throttle_reasons: ['sw_power_cap', 'hw_thermal', 'unknown_0x400'] });
        expect(byTestId('sys-throttle-chip')?.title).toBe('hw_thermal');
    });

    it('takes the chip away when the reason clears', () => {
        show({ throttle_reasons: ['hw_thermal'] });
        expect(byTestId('sys-throttle-chip')).not.toBeNull();
        show({ throttle_reasons: [] });
        expect(byTestId('sys-throttle-chip')).toBeNull();
    });

    it('says a card that stopped answering is unreadable, with the error in the tooltip', () => {
        show({
            readable: false,
            health_error: 'GPU is lost',
            temperature_margin_c: null,
            fan_percent: null,
            throttle_reasons: null,
        });
        const chip = byTestId('sys-unreadable-chip');
        expect(chip?.textContent).toContain('Card unreadable');
        expect(chip?.title).toBe('GPU is lost');
        expect(byTestId('sys-gpu-health')).toBeNull();
        expect(byTestId('sys-throttle-chip')).toBeNull();
    });

    it('does not call a healthy card unreadable', () => {
        show({ readable: true, temperature_margin_c: 12, fan_percent: 64, throttle_reasons: [] });
        expect(byTestId('sys-unreadable-chip')).toBeNull();
    });
});
