import { defineConfig } from 'vitest/config';

// Runner options for @angular/build:unit-test (angular.json "runnerConfig").
// vitest 5 forwards every console call from a worker to the main process over an rpc
// ("onUserConsoleLog"); a call still in flight when the worker closes is raised as an
// unhandled EnvironmentTeardownError that fails an otherwise green run. Writing the
// console straight to the stream removes the rpc, so there is nothing to be pending.
export default defineConfig({
  test: {
    disableConsoleIntercept: true,
  },
});
