import 'zone.js/testing';
import { getTestBed } from '@angular/core/testing';
import {
    BrowserDynamicTestingModule,
    platformBrowserDynamicTesting
} from '@angular/platform-browser-dynamic/testing';

// First, initialize the Angular testing environment.
getTestBed().initTestEnvironment(
    BrowserDynamicTestingModule,
    platformBrowserDynamicTesting()
);
// Inert WebSocket so the global WebSocketService (constructed in its ctor via
// connect()) never opens a real socket and never enters its 1s reconnect loop
// under Karma. Without this, the loop keeps the browser process alive so the
// run never exits and istanbul teardown is blocked.
class MockWebSocket {
    static readonly CONNECTING = 0;
    static readonly OPEN = 1;
    static readonly CLOSING = 2;
    static readonly CLOSED = 3;
    readyState = MockWebSocket.CONNECTING;
    onopen: ((e: any) => void) | null = null;
    onmessage: ((e: any) => void) | null = null;
    onclose: ((e: any) => void) | null = null;
    onerror: ((e: any) => void) | null = null;
    constructor(public url: string) { /* never connects */ }
    send(_data: any): void { /* no-op */ }
    close(): void { this.readyState = MockWebSocket.CLOSED; }
}
(window as any).WebSocket = MockWebSocket;

// Unfortunately there's no typing for the `__karma__` variable. Just declare it as any.
declare const __karma__: any;
declare const require: any;

// Prevent Karma from running prematurely.
__karma__.loaded = function () { };

// First, find all the tests.
const context = require.context('./', true, /\.spec\.ts$/);
// And load the modules.
context.keys().map(context);
// Finally, start Karma to run the tests.
__karma__.start();
