import { Component, OnInit, OnDestroy, ElementRef, viewChild, signal, inject, effect } from '@angular/core';
import { JobService, JobStatus } from '../../../services/job';
import { Subscription, interval, switchMap, startWith, of, catchError } from 'rxjs';

@Component({
  selector: 'app-live-terminal',
  standalone: true,
  imports: [],
  template: `
    <div class="bg-overlay text-brand font-mono p-4 rounded-theme-xl h-96 overflow-y-auto shadow-2xl border border-surface-mid scroll-smooth backdrop-blur-md" 
         data-testid="live-terminal-container"
         #terminalContainer>
      <div class="flex flex-col gap-1">
        @for (line of logs(); track $index) {
          <div class="flex gap-3 items-start group" data-testid="terminal-line">
            <span class="text-text-disabled text-[10px] select-none mt-1 min-w-[20px]">{{ $index + 1 }}</span>
            <div [class.text-success]="isJson(line)" 
                 [class.text-danger]="line.toLowerCase().includes('error')"
                 [class.text-warning]="line.toLowerCase().includes('warning')"
                 class="whitespace-pre-wrap text-xs leading-relaxed font-medium break-all selection:bg-brand/30">
              {{ formatLine(line) }}
            </div>
          </div>
        }
        @if (logs().length === 0) {
          <div class="flex flex-col items-center justify-center h-full text-text-disabled gap-3" data-testid="terminal-empty">
            <div class="w-12 h-12 border-2 border-surface-mid border-t-brand rounded-full animate-spin"></div>
            <div class="italic text-sm tracking-wide">Awaiting training directives...</div>
          </div>
        }
      </div>
    </div>
  `,
  styles: []
})
export class LiveTerminalComponent implements OnInit, OnDestroy {
  logs = signal<string[]>([]);
  private jobService = inject(JobService);
  private sub?: Subscription;

  terminalContainer = viewChild<ElementRef>('terminalContainer');

  constructor() {
    effect(() => {
      const logs = this.logs();
      if (logs.length > 0) {
        this.scrollToBottom();
      }
    });
  }

  ngOnInit() {
    this.startPolling();
  }

  startPolling() {
    this.sub = interval(3000).pipe(
      startWith(0),
      switchMap(() => this.jobService.listJobs().pipe(
        catchError(err => {
          console.error('Error listing jobs:', err);
          return of([]);
        })
      )),
      switchMap(jobs => {
        const runningJob = jobs.find(j => j.status === JobStatus.RUNNING);
        if (runningJob) {
          return this.jobService.getJobLogs(runningJob.id).pipe(
            catchError(err => {
              console.error('Error fetching logs:', err);
              return of(this.logs());
            })
          );
        }
        return of(this.logs()); // Keep current logs if nothing running
      })
    ).subscribe({
      next: (logs) => {
        this.logs.set(logs);
      },
      error: (err) => console.error('LiveTerminal polling failed (should be caught by catchError):', err)
    });
  }

  isJson(line: string): boolean {
    return line.trim().startsWith('{') && line.trim().endsWith('}');
  }

  formatLine(line: string): string {
    if (this.isJson(line)) {
      try {
        const obj = JSON.parse(line);
        if (obj.status === 'training') {
          return `[TRAIN] Step: ${obj.step} | Loss: ${obj.loss.toFixed(6)} | Progress: ${obj.progress}%`;
        }
        return JSON.stringify(obj, null, 2);
      } catch (e) {
        return line;
      }
    }
    return line;
  }

  scrollToBottom() {
    setTimeout(() => {
      const el = this.terminalContainer();
      if (el) {
        el.nativeElement.scrollTop = el.nativeElement.scrollHeight;
      }
    }, 100);
  }

  ngOnDestroy() {
    this.sub?.unsubscribe();
  }
}
