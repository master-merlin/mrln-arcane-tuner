import { Routes } from '@angular/router';

export const routes: Routes = [
    { path: '', pathMatch: 'full', redirectTo: 'datasets' },
    {
        path: 'datasets',
        loadComponent: () =>
            import('./screens/datasets-screen/datasets-screen').then(m => m.DatasetsScreen),
    },
    {
        path: 'projects',
        loadComponent: () =>
            import('./screens/projects-screen/projects-screen').then(m => m.ProjectsScreen),
    },
    {
        path: 'projects/:id',
        loadComponent: () =>
            import('./screens/projects-screen/project-detail').then(m => m.ProjectDetail),
    },
    {
        path: 'training',
        loadComponent: () =>
            import('./screens/training-screen/training-screen').then(m => m.TrainingScreen),
    },
    {
        path: 'jobs',
        loadComponent: () =>
            import('./screens/jobs-screen/jobs-screen').then(m => m.JobsScreen),
    },
    {
        path: 'tools',
        loadComponent: () =>
            import('./screens/tools-screen/tools-screen').then(m => m.ToolsScreen),
    },
    {
        path: 'server',
        loadComponent: () =>
            import('./screens/server-screen/server-screen').then(m => m.ServerScreen),
    },
    { path: '**', redirectTo: 'datasets' },
];
