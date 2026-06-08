import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { RuntimeConfigService } from './runtime-config.service';

/** The header a peek returns; ``kind`` routes the import to the right flow. */
export interface ArchivePeek {
    kind: 'dataset' | 'template' | 'project';
    format_version?: number;
    app_version?: string;
}

@Injectable({ providedIn: 'root' })
export class ImportArchiveService {
    private http = inject(HttpClient);
    private rtc = inject(RuntimeConfigService);

    /** Read a dropped archive's manifest header to learn its ``kind``. */
    peekImport(file: File): Observable<ArchivePeek> {
        const form = new FormData();
        form.append('file', file);
        return this.http.post<ArchivePeek>(`${this.rtc.apiUrl}/import/peek`, form);
    }

    /** Save a blob to disk via a transient anchor (revokes the object URL). */
    downloadBlob(blob: Blob, filename: string): void {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();
        URL.revokeObjectURL(url);
    }
}
