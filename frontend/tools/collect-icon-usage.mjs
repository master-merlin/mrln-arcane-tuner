/**
 * Print every Lucide icon name this app can ask for, as JSON.
 *
 * A STARTING POINT, NOT AN ORACLE. `src/app/icons/icon-set.ts` is the list
 * that ships, and the TypeScript compiler is what proves it complete: because
 * `IconKey` is `keyof typeof iconSet`, a name that is used but missing is a
 * build error at the call site. This script exists so that adding a screenful
 * of icons does not mean hand-tracing call sites, and so that pruning has a
 * defensible starting set.
 *
 * It was written by running it against the real tree and then building: it
 * missed three names on its first pass (`ExternalLink`, `Sun`, `Moon`, all
 * returned from helper methods) and the compiler named all three. Pass D below
 * is the widening that fixed that. Treat a future miss the same way — widen
 * the pass, do not widen `IconKey`.
 *
 *   node tools/collect-icon-usage.mjs
 *
 * Deliberately NOT wired into the build. A generator that runs on every build
 * would silently add whatever the scanner happened to match, which is how a
 * curated list stops being curated.
 */
import fs from 'node:fs';
import path from 'node:path';

const ROOT = process.cwd();
const SRC = path.join(ROOT, 'src');
const DTS = path.join(
    ROOT, 'node_modules', '@lucide', 'angular', 'types', 'lucide-angular.d.ts',
);

const dts = fs.readFileSync(DTS, 'utf8');
const exported = new Set(
    dts.slice(dts.lastIndexOf('\nexport {')).match(/Lucide[A-Za-z0-9]+/g) ?? [],
);

function walk(dir, out = []) {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
        const p = path.join(dir, entry.name);
        if (entry.isDirectory()) walk(p, out);
        else if (/\.(ts|html)$/.test(entry.name) && !/\.spec\.ts$/.test(entry.name)) out.push(p);
    }
    return out;
}

const found = new Map();
const add = (name, pass) => {
    if (!name || !exported.has(`Lucide${name}`)) return;
    if (!found.has(name)) found.set(name, new Set());
    found.get(name).add(pass);
};

for (const file of walk(SRC)) {
    const text = fs.readFileSync(file, 'utf8');

    // A — static attribute: <app-ico name="Database">
    for (const m of text.matchAll(/<app-ico\b[^>]*?\sname="([A-Za-z0-9]+)"/g)) add(m[1], 'A');

    // B — literals inside a bound expression: [name]="cond ? 'Minus' : 'X'"
    for (const m of text.matchAll(/\[name\]="([^"]*)"/g)) {
        for (const lit of m[1].matchAll(/'([A-Za-z0-9]+)'/g)) add(lit[1], 'B');
    }
    for (const attr of ['icon', 'ico', 'iconName']) {
        for (const m of text.matchAll(new RegExp(`\\[${attr}\\]="([^"]*)"`, 'g'))) {
            for (const lit of m[1].matchAll(/'([A-Za-z0-9]+)'/g)) add(lit[1], 'B');
        }
        for (const m of text.matchAll(new RegExp(`\\s${attr}="([A-Za-z0-9]+)"`, 'g'))) add(m[1], 'A');
    }

    // C — data tables: { icon: 'Play', label: … }
    for (const m of text.matchAll(/\b(?:icon|ico|iconName)\s*:\s*['"]([A-Za-z0-9]+)['"]/g)) {
        add(m[1], 'C');
    }

    // D — helpers that COMPUTE a name (themeIcon(), editIcon(domain)). Scoped
    // to files that deal in icons, and to the two shapes such a helper takes.
    // Widening this to "every PascalCase literal" would vacuum up ordinary
    // words that happen to be icon names: Image, Type, Server, History, Box.
    const dealsInIcons =
        text.includes('IconKey') || text.includes('IcoComponent') || text.includes('app-ico');
    if (dealsInIcons) {
        for (const m of text.matchAll(/IconKey\s*=\s*'([A-Za-z0-9]+)'/g)) add(m[1], 'D');
        for (const m of text.matchAll(/return\s+'([A-Za-z0-9]+)'/g)) add(m[1], 'D');
        for (const m of text.matchAll(/\?\s*'([A-Za-z0-9]+)'\s*:\s*'([A-Za-z0-9]+)'/g)) {
            add(m[1], 'D');
            add(m[2], 'D');
        }
    }
}

const names = [...found.keys()].sort();
console.log(JSON.stringify({ count: names.length, names }));
