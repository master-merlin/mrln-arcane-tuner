import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { RuntimeConfigService } from './runtime-config.service';

/** `GET /filesystem/browse` — directory listing for a path-picker dialog. */
export interface BrowseResponse {
  path: string;
  parent: string;
  entries: { name: string; path: string; type: string }[];
}

/**
 * Server-side filesystem access shared by any path-entry UI (dynamic training
 * config fields, server settings' default model path, …). Two distinct
 * backend affordances live here: a lightweight directory-listing endpoint
 * (`browse`, used to render an in-page folder tree) and a native OS folder
 * picker dialog (`pickFolder`).
 */
@Injectable({ providedIn: 'root' })
export class FilesystemService {
  private http = inject(HttpClient);
  private apiUrl = inject(RuntimeConfigService).apiUrl;

  /** List a directory's immediate contents (+ parent) for the in-page browse dropdown. */
  browse(path: string): Observable<BrowseResponse> {
    return this.http.get<BrowseResponse>(`${this.apiUrl}/filesystem/browse`, { params: { path } });
  }

  /** Open a native OS folder picker dialog via the backend. */
  pickFolder(initialDir: string, title: string): Observable<{ path: string }> {
    return this.http.post<{ path: string }>(
      `${this.apiUrl}/filesystem/pick-folder`,
      { initial_dir: initialDir, title },
    );
  }
}
