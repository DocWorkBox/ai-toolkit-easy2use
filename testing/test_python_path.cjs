const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');
const ts = require('../ui/node_modules/typescript');

const source = fs.readFileSync(path.join(__dirname, '../ui/cron/pythonPath.ts'), 'utf8');
const compiled = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, esModuleInterop: true },
}).outputText;

function resolver(files, platform = 'win32', env = {}) {
  const paths = platform === 'win32' ? path.win32 : path.posix;
  const root = platform === 'win32' ? 'C:\\portable toolkit' : '/toolkit';
  const existing = new Set(files.map(file => paths.join(root, file)));
  const exports = {};
  vm.runInNewContext(compiled, {
    exports,
    process: { platform, env },
    require(name) {
      if (name === 'path') return paths;
      if (name === 'fs') return { existsSync: file => existing.has(file) };
      if (name === './paths') return { TOOLKIT_ROOT: root };
      throw new Error(`Unexpected import: ${name}`);
    },
  });
  return { ...exports, root, paths };
}

test('portable UI jobs use the bundled pythonw with an absolute path', () => {
  const r = resolver(['runtime/python/python.exe', 'runtime/python/pythonw.exe']);
  assert.equal(r.resolvePythonPath(), r.paths.join(r.root, 'runtime/python/python.exe'));
  assert.equal(r.resolveDetachedPythonPath(), r.paths.join(r.root, 'runtime/python/pythonw.exe'));
});

test('portable detection matches the manager when a venv is also present', () => {
  const r = resolver(['runtime/python/python.exe', '.venv/Scripts/python.exe']);
  assert.equal(r.resolvePythonPath(), r.paths.join(r.root, 'runtime/python/python.exe'));
});

test('an explicit standard layout retains the venv interpreter', () => {
  const r = resolver(
    ['runtime/python/python.exe', '.venv/Scripts/python.exe', '.venv/Scripts/pythonw.exe'],
    'win32', { AITK_RUNTIME_LAYOUT: 'standard' },
  );
  assert.equal(r.resolveDetachedPythonPath(), r.paths.join(r.root, '.venv/Scripts/pythonw.exe'));
});

test('portable installs without pythonw retain the bundled console interpreter', () => {
  const r = resolver(['runtime/python/python.exe']);
  assert.equal(r.resolveDetachedPythonPath(), r.paths.join(r.root, 'runtime/python/python.exe'));
});

test('standard Windows and Linux installations retain their venv resolution', () => {
  const windows = resolver(['venv/Scripts/python.exe', 'venv/Scripts/pythonw.exe']);
  assert.equal(windows.resolveDetachedPythonPath(), windows.paths.join(windows.root, 'venv/Scripts/pythonw.exe'));
  const linux = resolver(['.venv/bin/python'], 'linux');
  assert.equal(linux.resolveDetachedPythonPath(), '/toolkit/.venv/bin/python');
});
