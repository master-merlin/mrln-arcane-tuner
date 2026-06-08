import { describe, it, expect } from 'vitest';
import { DatasetFormModalComponent } from './dataset-form.component';

const DATASETS = [
    { id: '1', name: 'Portraits' },
    { id: '2', name: 'Car v2' },
];

describe('DatasetFormModalComponent.nameCollides', () => {
    it('flags a name matching another dataset', () => {
        expect(DatasetFormModalComponent.nameCollides('Portraits', DATASETS, null)).toBe(true);
    });

    it('compares the SANITIZED name (parens stripped, matching the folder)', () => {
        // "Car (v2)" -> "Car v2", which already exists.
        expect(DatasetFormModalComponent.nameCollides('Car (v2)', DATASETS, null)).toBe(true);
    });

    it('does not flag the dataset being edited (keeping its own name)', () => {
        expect(DatasetFormModalComponent.nameCollides('Portraits', DATASETS, '1')).toBe(false);
    });

    it('does not flag a free name', () => {
        expect(DatasetFormModalComponent.nameCollides('Brand New', DATASETS, null)).toBe(false);
    });

    it('ignores an empty / all-stripped name (handled by the required validator)', () => {
        expect(DatasetFormModalComponent.nameCollides('   ', DATASETS, null)).toBe(false);
        expect(DatasetFormModalComponent.nameCollides('()!!', DATASETS, null)).toBe(false);
    });
});
