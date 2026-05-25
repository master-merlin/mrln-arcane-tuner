import { Component, input } from '@angular/core';

@Component({
    selector: 'app-dataset-empty-state',
    standalone: true,
    imports: [],
    template: `
    <div class="col-span-full py-20 text-center text-text-subtle border-2 border-dashed border-surface-mid rounded-theme-xl bg-surface-low/10 flex flex-col items-center gap-4 animate-in fade-in zoom-in duration-500">
        <div class="p-6 bg-surface-mid/20 rounded-full border border-surface-high/30">
            <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" stroke-linecap="round" stroke-linejoin="round" class="text-text-disabled">
                <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path>
                <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path>
            </svg>
        </div>
        
        <div class="space-y-1">
            <p class="text-lg font-semibold text-text-muted">
                @if (searchTerm()) {
                    No datasets match "{{ searchTerm() }}"
                } @else {
                    Your library is empty
                }
            </p>
            <p class="text-sm text-text-disabled max-w-xs mx-auto">
                @if (searchTerm()) {
                    Try a different search term or clear the filter to see all datasets.
                } @else {
                    Click "New Dataset" to start organizing your training data.
                }
            </p>
        </div>
    </div>
  `,
    styles: []
})
export class DatasetEmptyStateComponent {
    searchTerm = input<string>('');
}
