import { Component, inject } from '@angular/core';
import { ToastService, Toast } from '../../../services/toast';

@Component({
  selector: 'app-toast-container',
  standalone: true,
  template: `
    <div class="fixed top-20 right-6 z-[10000] flex flex-col gap-2.5 pointer-events-none max-w-md w-full">
      @for (toast of toastService.toasts(); track toast.id) {
        <div
          (click)="toastService.dismiss(toast.id)"
          class="pointer-events-auto flex items-start gap-3.5 px-5 py-3.5 rounded-theme-lg shadow-2xl backdrop-blur-xl border cursor-pointer transition-all duration-300 animate-slideIn"
          [class]="toastClass(toast)">
          <span class="mt-0.5 shrink-0" [innerHTML]="toastIcon(toast)"></span>
          <span class="text-[15px] font-medium leading-snug break-words">{{ toast.message }}</span>
        </div>
      }
    </div>
  `,
  styles: [`
    @keyframes slideIn {
      from { opacity: 0; transform: translateX(1rem); }
      to   { opacity: 1; transform: translateX(0); }
    }
    :host ::ng-deep .animate-slideIn {
      animation: slideIn 0.25s ease-out;
    }
  `]
})
export class ToastContainerComponent {
  toastService = inject(ToastService);

  toastClass(toast: Toast): string {
    const base = 'bg-surface-low/90 text-text-primary';
    switch (toast.type) {
      case 'success': return `${base} border-success/40`;
      case 'error': return `${base} border-danger/40`;
      case 'warning': return `${base} border-warning/40`;
      case 'info': return `${base} border-brand/40`;
    }
  }

  toastIcon(toast: Toast): string {
    const size = 'width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"';
    switch (toast.type) {
      case 'success': return `<svg xmlns="http://www.w3.org/2000/svg" ${size} class="text-success"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>`;
      case 'error': return `<svg xmlns="http://www.w3.org/2000/svg" ${size} class="text-danger"><circle cx="12" cy="12" r="10"></circle><line x1="15" y1="9" x2="9" y2="15"></line><line x1="9" y1="9" x2="15" y2="15"></line></svg>`;
      case 'warning': return `<svg xmlns="http://www.w3.org/2000/svg" ${size} class="text-warning"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>`;
      case 'info': return `<svg xmlns="http://www.w3.org/2000/svg" ${size} class="text-brand"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>`;
    }
  }
}
