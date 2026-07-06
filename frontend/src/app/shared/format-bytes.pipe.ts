import { Pipe, PipeTransform } from '@angular/core';
import { formatBytes } from './format-bytes';

/** Template wrapper around {@link formatBytes} — `{{ n | formatBytes }}`.
 *  Replaces the character-identical `protected formatBytes()` wrappers that
 *  used to live on `DatasetsScreen` and `JobsScreen` for template access. */
@Pipe({ name: 'formatBytes', standalone: true })
export class FormatBytesPipe implements PipeTransform {
    transform(n: number): string {
        return formatBytes(n);
    }
}
