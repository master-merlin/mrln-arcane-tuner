import { DebugElement } from '@angular/core';
import { ComponentFixture } from '@angular/core/testing';
import { By } from '@angular/platform-browser';

/**
 * Canonical C6 selector: query a single element by its data-testid.
 * Returns `null` when no element matches (mirrors `DebugElement.query`).
 */
export function byTestId<T>(fixture: ComponentFixture<T>, id: string): DebugElement | null {
  return fixture.debugElement.query(By.css(`[data-testid="${id}"]`));
}

/** Query all elements matching a data-testid (e.g. dynamic list rows). */
export function allByTestId<T>(fixture: ComponentFixture<T>, id: string): DebugElement[] {
  return fixture.debugElement.queryAll(By.css(`[data-testid="${id}"]`));
}
