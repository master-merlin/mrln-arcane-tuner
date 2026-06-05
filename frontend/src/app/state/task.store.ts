import { Injectable, computed, effect, signal } from '@angular/core';
import { WebSocketService } from '../services/websocket.service';
import { DatasetService } from '../services/dataset';

export type TaskStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';

export interface Task {
    id: string;
    type: string;
    title: string;
    status: TaskStatus;
    dataset_name: string | null;
    total: number;
    current: number;
    current_item: string | null;
    ok: number;
    failed: number;
    created_at: number;
    started_at: number | null;
    finished_at: number | null;
    error: string | null;
}

export interface RecentTask extends Task { recordedAt: number; }

const RECENT_CAP = 8;
const RECENT_TTL_MS = 5 * 60 * 1000;
const ACTIVE: TaskStatus[] = ['pending', 'running'];

@Injectable({ providedIn: 'root' })
export class TaskStore {
    private _active = signal<Map<string, Task>>(new Map());
    private _recent = signal<RecentTask[]>([]);

    readonly active = computed<Task[]>(() => Array.from(this._active().values()));
    readonly activeCount = computed(() => this._active().size);
    readonly recent = computed<RecentTask[]>(() => {
        const cutoff = Date.now() - RECENT_TTL_MS;
        return this._recent().filter(r => r.recordedAt >= cutoff);
    });

    byId(id: string) {
        return computed<Task | undefined>(() => this._active().get(id)
            ?? this._recent().find(r => r.id === id));
    }

    constructor(private ws: WebSocketService, private api: DatasetService) {
        this.ws.on<Task>('task_update').subscribe(t => this.apply(t));
        // Re-sync on first load and on every reconnect (server-side tasks
        // survive client reload).
        effect(() => { this.ws.reconnected(); this.resync(); });
    }

    private resync(): void {
        this.api.getTasks().subscribe((tasks: Task[]) => {
            for (const t of tasks ?? []) this.apply(t);
        });
    }

    private apply(t: Task): void {
        if (ACTIVE.includes(t.status)) {
            this._active.update(m => new Map(m).set(t.id, t));
            return;
        }
        this._active.update(m => { const n = new Map(m); n.delete(t.id); return n; });
        this._recent.update(r => [
            { ...t, recordedAt: Date.now() },
            ...r.filter(x => x.id !== t.id),
        ].slice(0, RECENT_CAP));
    }

    cancel(id: string): void {
        this.api.cancelTask(id).subscribe({ error: () => undefined });
    }
}
