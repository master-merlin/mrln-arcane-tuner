import { Injectable, signal } from '@angular/core';

export type ToastType = 'success' | 'error' | 'warning' | 'info';

export interface Toast {
    id: number;
    type: ToastType;
    message: string;
}

export interface ToastHistoryEntry {
    id: number;
    type: ToastType;
    message: string;
    timestamp: number; // epoch ms, captured when the toast is shown
}

const DURATIONS: Record<ToastType, number> = {
    success: 3000,
    info: 4000,
    warning: 5000,
    error: 6000,
};

const HISTORY_CAP = 20;

@Injectable({ providedIn: 'root' })
export class ToastService {
    private nextId = 0;
    readonly toasts = signal<Toast[]>([]);
    /** Newest-first record of the last HISTORY_CAP toasts shown this session. */
    readonly history = signal<ToastHistoryEntry[]>([]);

    success(message: string, duration?: number) { this.show('success', message, duration); }
    error(message: string, duration?: number) { this.show('error', message, duration); }
    warning(message: string, duration?: number) { this.show('warning', message, duration); }
    info(message: string, duration?: number) { this.show('info', message, duration); }

    dismiss(id: number) {
        this.toasts.update(list => list.filter(t => t.id !== id));
    }

    private show(type: ToastType, message: string, duration?: number) {
        const id = this.nextId++;
        this.toasts.update(list => [...list, { id, type, message }]);
        this.history.update(list =>
            [{ id, type, message, timestamp: Date.now() }, ...list].slice(0, HISTORY_CAP),
        );
        setTimeout(() => this.dismiss(id), duration ?? DURATIONS[type]);
    }
}
