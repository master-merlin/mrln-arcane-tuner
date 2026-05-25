import { Component, input, output, signal, effect, computed } from '@angular/core';
import { TitleCasePipe } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Dataset } from '../../../../services/dataset';

@Component({
    selector: 'app-dataset-form-modal',
    standalone: true,
    imports: [TitleCasePipe, FormsModule],
    template: `
    <div class="fixed inset-0 bg-overlay backdrop-blur-md z-[100] flex items-center justify-center p-4 animate-in fade-in duration-300" (click)="cancel.emit()">
        <div class="bg-surface-low border border-surface-mid rounded-theme-xl w-full max-w-md shadow-2xl p-8 transform animate-in slide-in-from-bottom-4 duration-300" (click)="$event.stopPropagation()">
            
            <div class="flex items-center gap-4 mb-6">
                <div class="p-3 bg-brand/10 rounded-theme-md border border-brand/20 text-brand">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path>
                        <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path>
                    </svg>
                </div>
                <div>
                   <h3 class="text-xl font-bold text-white">{{ isEditing() ? 'Update Dataset' : 'Add New Dataset' }}</h3>
                   <p class="text-xs text-text-subtle mt-0.5">Define your dataset metadata for better organization.</p>
                </div>
            </div>
            
            <div class="space-y-5">
                <!-- Name -->
                <div>
                    <label class="block text-[10px] font-black uppercase tracking-widest text-text-subtle mb-1.5 ml-1">Dataset Name</label>
                    <input 
                        [(ngModel)]="name" 
                        data-testid="input-dataset-name"
                        [class.border-danger]="isNameInvalid()"
                        [class.focus:ring-danger/20]="isNameInvalid()"
                        type="text" 
                        class="w-full bg-base/60 border border-surface-mid rounded-theme-md px-4 py-3 text-white focus:outline-none focus:ring-2 focus:ring-brand/20 focus:border-brand transition-all font-medium" 
                        placeholder="e.g. MyAmazingLoRASet"
                    >
                    @if (isNameInvalid()) {
                        <div class="text-danger text-[10px] mt-1.5 ml-1 flex items-center gap-1">
                           <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>
                           Name contains forbidden characters (< > : " / \\ | ? *)
                        </div>
                    }
                </div>
                
                <!-- Classifier -->
                <div class="grid grid-cols-1 gap-4">
                    <div>
                        <label class="block text-[10px] font-black uppercase tracking-widest text-text-subtle mb-1.5 ml-1">Category / Classifier</label>
                        <div class="flex flex-col gap-3">
                            <select 
                                [(ngModel)]="classifierSelection" 
                                data-testid="select-classifier"
                                class="w-full bg-base/60 border border-surface-mid rounded-theme-md px-4 py-3 text-white focus:outline-none focus:ring-2 focus:ring-brand/20 focus:border-brand transition-all appearance-none cursor-pointer"
                            >
                                <option value="">None / Uncategorized</option>
                                <option disabled>── Standard Categories ──</option>
                                @for (std of standardClassifiers; track std) {
                                    <option [value]="std">{{ std | titlecase }}</option>
                                }
                                
                                @if (reusableClassifiers().length > 0) {
                                    <option disabled>── Previously Used ──</option>
                                    @for (c of reusableClassifiers(); track c) {
                                        <option [value]="c">{{ c }}</option>
                                    }
                                }
                                <option disabled>──────────</option>
                                <option value="custom">Create New Category...</option>
                            </select>

                            @if (classifierSelection() === 'custom') {
                                <input 
                                    [(ngModel)]="customClassifierValue" 
                                    data-testid="input-custom-classifier"
                                    type="text" 
                                    class="w-full bg-brand/5 border border-brand/30 rounded-theme-md px-4 py-3 text-white placeholder-gray-600 focus:outline-none focus:ring-2 focus:ring-brand/20 focus:border-brand transition-all font-medium animate-in slide-in-from-top-2 duration-300" 
                                    placeholder="Enter your custom category name..."
                                >
                            }
                        </div>
                    </div>
                </div>

                <!-- Description -->
                <div>
                    <label class="block text-[10px] font-black uppercase tracking-widest text-text-subtle mb-1.5 ml-1">Description</label>
                    <textarea [(ngModel)]="description" 
                        data-testid="input-dataset-description"
                        rows="3" 
                        class="w-full bg-base/60 border border-surface-mid rounded-theme-md px-4 py-3 text-white focus:outline-none focus:ring-2 focus:ring-brand/20 focus:border-brand transition-all resize-none font-medium" 
                        placeholder="Optional description for this dataset..."></textarea>
                </div>
            </div>

            <!-- Footer -->
            <div class="flex flex-col md:flex-row justify-end gap-3 mt-8">
                <button (click)="cancel.emit()" 
                        class="order-2 md:order-1 text-text-subtle hover:text-white text-sm font-bold px-6 py-3 transition-colors uppercase tracking-widest">
                    Cancel
                </button>
                <button 
                    (click)="submit()" 
                    data-testid="btn-save-dataset"
                    [disabled]="!name() || isNameInvalid()" 
                    class="order-1 md:order-2 bg-gradient-to-r from-brand to-brand-gradient-end hover:from-brand/90 hover:to-brand-gradient-end disabled:opacity-30 disabled:cursor-not-allowed text-white px-8 py-3 rounded-theme-lg text-sm font-black uppercase tracking-widest transition-all shadow-xl shadow-brand/20 active:scale-95"
                >
                    {{ isEditing() ? 'Save Changes' : 'Initialize Dataset' }}
                </button>
            </div>
        </div>
    </div>
  `,
    styles: []
})
export class DatasetFormModalComponent {
    dataset = input<Dataset | null>(null);
    reusableClassifiers = input<string[]>([]);

    save = output<{ name: string, description: string, classifier: string }>();
    cancel = output<void>();

    standardClassifiers = ['vehicle', 'person', 'style', 'object', 'landscape'];

    // Internal Signals
    name = signal('');
    description = signal('');
    classifierSelection = signal('');
    customClassifierValue = signal('');
    isEditing = computed(() => !!this.dataset());

    constructor() {
        effect(() => {
            const ds = this.dataset();
            if (ds) {
                this.name.set(ds.name);
                this.description.set(ds.description || '');

                const classifier = ds.classifier || '';
                const standard = [...this.standardClassifiers, ''];

                if (classifier && !standard.includes(classifier.toLowerCase())) {
                    if (this.reusableClassifiers().includes(classifier)) {
                        this.classifierSelection.set(classifier);
                        this.customClassifierValue.set('');
                    } else {
                        this.classifierSelection.set('custom');
                        this.customClassifierValue.set(classifier);
                    }
                } else {
                    this.classifierSelection.set(classifier);
                    this.customClassifierValue.set('');
                }
            } else {
                this.name.set('');
                this.description.set('');
                this.classifierSelection.set('');
                this.customClassifierValue.set('');
            }
        }, { allowSignalWrites: true });
    }

    isNameInvalid(): boolean {
        const forbiddenChars = /[<>:"/\\|?*]/;
        return forbiddenChars.test(this.name());
    }

    submit() {
        if (!this.name() || this.isNameInvalid()) return;

        const classifier = this.classifierSelection() === 'custom'
            ? this.customClassifierValue()
            : this.classifierSelection();

        this.save.emit({
            name: this.name(),
            description: this.description(),
            classifier
        });
    }
}
