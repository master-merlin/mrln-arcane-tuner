import { describe, it, expect, vi } from 'vitest';
import { of } from 'rxjs';
import { ProjectsScreen } from './projects-screen';
import type { Project, ProjectStats } from '../../services/project.service';

interface ConfirmData {
  destructive?: boolean;
  onConfirm?: () => void;
}

const STATS = (over: Partial<ProjectStats> = {}): ProjectStats => ({
  captioning_templates: 0, masking_templates: 0, training_templates: 0,
  datasets: 0, jobs: 0, ...over,
});
const PROJ = (over: Partial<Project> = {}): Project => ({
  id: 'p', name: 'P', description: '', color: '#000',
  created_at: 0, updated_at: 0, ...over,
});

describe('ProjectsScreen.cardStat — absent stats render em-dash, not 0 (P2)', () => {
  it('returns null for every key when the project has no stats block', () => {
    expect(ProjectsScreen.cardStat(undefined, 'datasets')).toBeNull();
    expect(ProjectsScreen.cardStat(undefined, 'templates')).toBeNull();
    expect(ProjectsScreen.cardStat(undefined, 'jobs')).toBeNull();
  });

  it('returns real numbers (including 0) when stats are present', () => {
    const s = STATS({ datasets: 3, jobs: 0, captioning_templates: 1, masking_templates: 2, training_templates: 4 });
    expect(ProjectsScreen.cardStat(s, 'datasets')).toBe(3);
    expect(ProjectsScreen.cardStat(s, 'jobs')).toBe(0);
    expect(ProjectsScreen.cardStat(s, 'templates')).toBe(7); // 1 + 2 + 4
  });
});

describe('ProjectsScreen.aggregate — KPI rail (P2)', () => {
  it('returns null when projects exist but NONE carry stats', () => {
    const list = [PROJ({ id: 'a' }), PROJ({ id: 'b' })];
    expect(ProjectsScreen.aggregate(list, 'datasets')).toBeNull();
    expect(ProjectsScreen.aggregate(list, 'jobs')).toBeNull();
  });

  it('sums the projects that DO carry stats, treating absent ones as 0 contribution', () => {
    const list = [
      PROJ({ id: 'a', stats: STATS({ datasets: 2, jobs: 1 }) }),
      PROJ({ id: 'b' }), // no stats
      PROJ({ id: 'c', stats: STATS({ datasets: 3, jobs: 0 }) }),
    ];
    expect(ProjectsScreen.aggregate(list, 'datasets')).toBe(5);
    expect(ProjectsScreen.aggregate(list, 'jobs')).toBe(1);
  });

  it('returns 0 (not null) for an empty project list', () => {
    expect(ProjectsScreen.aggregate([], 'datasets')).toBe(0);
  });
});

function invoke(method: string, ctx: Record<string, unknown>, ...args: unknown[]): unknown {
  const proto = ProjectsScreen.prototype as unknown as Record<string, (...a: unknown[]) => unknown>;
  return proto[method].apply(ctx, args);
}

describe('ProjectsScreen.deleteProject — themed confirm', () => {
  it('opens the destructive confirm modal and does not delete until confirmed', () => {
    const deleteProject = vi.fn().mockReturnValue(of({}));
    const openModal = vi.fn();
    const ctx = {
      overlay: { openModal },
      projects: { deleteProject, loadProjects: vi.fn() },
      toast: { success: vi.fn(), error: vi.fn() },
      afterDeleteProject: vi.fn(),
    };

    invoke('deleteProject', ctx, { id: 'p1', name: 'Demo' }, { stopPropagation: vi.fn() });

    expect(openModal).toHaveBeenCalledWith('confirm', expect.objectContaining({ destructive: true }));
    expect(deleteProject).not.toHaveBeenCalled();

    const data = openModal.mock.calls.at(-1)![1] as ConfirmData;
    data.onConfirm!();
    expect(deleteProject).toHaveBeenCalledWith('p1');
  });
});
