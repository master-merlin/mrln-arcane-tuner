import { TestBed } from '@angular/core/testing';
import { signal } from '@angular/core';
import {
    ExportOptionsModalComponent,
    type ExportOptionsData,
    type ExportSelection,
} from './export-options.component';
import { OverlayStore } from '../../state/overlay.store';

function mount(data: ExportOptionsData) {
    const closeModal = vi.fn();
    const topModal = signal({ kind: 'export-options', data });
    TestBed.configureTestingModule({
        imports: [ExportOptionsModalComponent],
        providers: [{ provide: OverlayStore, useValue: { topModal, closeModal } }],
    });
    const fixture = TestBed.createComponent(ExportOptionsModalComponent);
    fixture.detectChanges();
    return { cmp: fixture.componentInstance as ExportOptionsModalComponent, closeModal };
}

describe('ExportOptionsModalComponent', () => {
    it('initializes checkbox state from data (pre-checked items)', () => {
        const { cmp } = mount({
            title: 'Export',
            groups: [{
                key: 'templates',
                label: 'Templates',
                items: [
                    { id: 't1', label: 'A', checked: true },
                    { id: 't2', label: 'B' },
                ],
            }],
            onExport: () => undefined,
        });
        expect(cmp.isChecked('templates', 't1')).toBe(true);
        expect(cmp.isChecked('templates', 't2')).toBe(false);
    });

    it('toggles a checkbox', () => {
        const { cmp } = mount({
            title: 'Export',
            groups: [{ key: 'g', label: 'G', items: [{ id: 'x', label: 'X' }] }],
            onExport: () => undefined,
        });
        expect(cmp.isChecked('g', 'x')).toBe(false);
        cmp.toggle('g', 'x');
        expect(cmp.isChecked('g', 'x')).toBe(true);
        cmp.toggle('g', 'x');
        expect(cmp.isChecked('g', 'x')).toBe(false);
    });

    it('select all / none toggles every item in a group', () => {
        const group = {
            key: 'g', label: 'G',
            items: [{ id: 'a', label: 'A' }, { id: 'b', label: 'B' }, { id: 'c', label: 'C' }],
        };
        const { cmp } = mount({ title: 'Export', groups: [group], onExport: () => undefined });
        expect(cmp.checkedCount('g')).toBe(0);
        cmp.setAll(group, true);
        expect(cmp.checkedCount('g')).toBe(3);
        expect(cmp.isChecked('g', 'b')).toBe(true);
        cmp.setAll(group, false);
        expect(cmp.checkedCount('g')).toBe(0);
    });

    it('summary counts selected templates and non-excluded datasets', () => {
        const { cmp } = mount({
            title: 'Export',
            groups: [{ key: 'g', label: 'G', items: [{ id: 'a', label: 'A', checked: true }, { id: 'b', label: 'B' }] }],
            datasets: [{ name: 'd1', mode: 'reference' }, { name: 'd2', mode: 'embed' }],
            onExport: () => undefined,
        });
        expect(cmp['summary']()).toBe('1 template · 2 datasets');
        cmp.setMode('d2', 'exclude');
        expect(cmp['summary']()).toBe('1 template · 1 dataset');
    });

    it('initializes and updates per-dataset mode (tri-state)', () => {
        const { cmp } = mount({
            title: 'Export',
            datasets: [
                { name: 'big', sizeLabel: '4.2 GB', mode: 'reference' },
                { name: 'small', mode: 'embed' },
            ],
            onExport: () => undefined,
        });
        expect(cmp.modeOf('big')).toBe('reference');
        expect(cmp.modeOf('small')).toBe('embed');
        cmp.setMode('big', 'embed');
        expect(cmp.modeOf('big')).toBe('embed');
    });

    it('emits the collected selection and closes on confirm', () => {
        let captured: ExportSelection | null = null;
        const { cmp, closeModal } = mount({
            title: 'Export',
            groups: [{
                key: 'templates', label: 'Templates',
                items: [{ id: 't1', label: 'A', checked: true }, { id: 't2', label: 'B' }],
            }],
            datasets: [{ name: 'ds', mode: 'reference' }],
            onExport: (sel) => (captured = sel),
        });
        cmp.toggle('templates', 't2');         // now t1 + t2 checked
        cmp.setMode('ds', 'exclude');
        cmp.confirm();
        expect(captured).toEqual({
            groups: { templates: ['t1', 't2'] },
            datasets: [{ name: 'ds', mode: 'exclude' }],
        });
        expect(closeModal).toHaveBeenCalled();
    });

    it('defaults a dataset with no mode to reference', () => {
        const { cmp } = mount({
            title: 'Export',
            datasets: [{ name: 'ds' } as never],
            onExport: () => undefined,
        });
        expect(cmp.modeOf('ds')).toBe('reference');
    });
});
