const assert = require('node:assert/strict');
const test = require('node:test');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { createRequire } = require('node:module');

const root = path.resolve(__dirname, '..');
process.chdir(path.join(root, 'ui'));
const uiRequire = createRequire(path.join(root, 'ui/package.json'));
const ts = uiRequire('typescript');

function harness(value) {
  const rows = value === undefined ? [] : [{ key: 'DATASETS_FOLDER', value }];
  const writes = [];
  const prisma = { settings: {
    findMany: async () => rows,
    findFirst: async () => rows[0],
    upsert: async args => writes.push(args),
  } };
  const cache = new Map();
  function load(relative) {
    if (cache.has(relative)) return cache.get(relative);
    const filename = path.join(root, 'ui', relative);
    assert.ok(fs.existsSync(filename), `Missing shared path module: ${relative}`);
    const source = ts.transpileModule(fs.readFileSync(filename, 'utf8'), {
      compilerOptions: { module: ts.ModuleKind.CommonJS, esModuleInterop: true, target: ts.ScriptTarget.ES2020 },
    }).outputText;
    const module = { exports: {} };
    const require = name => {
      if (name === '@/server/prisma' || name === './prisma') return { default: prisma, __esModule: true };
      if (name === 'next/server') return { NextResponse: { json: data => data } };
      if (name === '@/paths') return load('src/paths.ts');
      if (name === '@/server/settings') return load('src/server/settings.ts');
      if (name.startsWith('.')) {
        return load(path.relative(path.join(root, 'ui'), path.resolve(path.dirname(filename), name + '.ts')));
      }
      return uiRequire(name);
    };
    vm.runInNewContext(source, { module, exports: module.exports, require, process, console }, { filename });
    cache.set(relative, module.exports);
    return module.exports;
  }
  return { load, writes };
}

test('the settings default is portable rather than the build machine path', async () => {
  const { load } = harness();
  assert.equal(load('src/paths.ts').defaultDatasetsFolder, './datasets');
  const settings = await load('src/app/api/settings/route.ts').GET();
  assert.equal(settings.DATASETS_FOLDER, './datasets');
});

test('relative paths follow the toolkit root when copied between drives', () => {
  const { load } = harness();
  const helper = load('cron/datasetPaths.ts');
  for (const root of ['C:\\portable-build\\ai-toolkit-easy2use-portable', 'F:\\Training Tools\\portable']) {
    for (const value of [undefined, '', './datasets', 'datasets', '.\\datasets\\']) {
      assert.equal(helper.resolveDatasetsRoot(value, root, path.win32), path.win32.join(root, 'datasets'));
    }
    assert.equal(helper.resolveDatasetsRoot('./custom', root, path.win32), path.win32.join(root, 'custom'));
  }
});

test('external drives and UNC datasets retain their configured locations', () => {
  const helper = harness().load('cron/datasetPaths.ts');
  const root = 'F:\\portable';
  for (const value of ['E:\\training-data', '\\\\nas\\share\\datasets']) {
    assert.equal(helper.normalizeDatasetsSetting(value, root, path.win32), value);
    assert.equal(helper.resolveDatasetsRoot(value, root, path.win32), value);
  }
});

test('only the current root default is normalized, not arbitrary absolute paths', () => {
  const helper = harness().load('cron/datasetPaths.ts');
  assert.equal(helper.normalizeDatasetsSetting('c:\\Portable\\datasets\\', 'C:\\portable', path.win32), './datasets');
  assert.equal(helper.normalizeDatasetsSetting('E:\\datasets', 'C:\\portable', path.win32), 'E:\\datasets');
});

test('settings GET and POST keep the default relative', async () => {
  const defaults = harness().load('src/paths.ts');
  const { load, writes } = harness(path.join(defaults.TOOLKIT_ROOT, 'datasets'));
  const route = load('src/app/api/settings/route.ts');
  assert.equal((await route.GET()).DATASETS_FOLDER, './datasets');
  await route.POST({ json: async () => ({ DATASETS_FOLDER: path.join(defaults.TOOLKIT_ROOT, 'datasets'),
                                         HF_TOKEN: '', TRAINING_FOLDER: '', MODELS_PATH: '' }) });
  const saved = writes.find(row => row.where.key === 'DATASETS_FOLDER');
  assert.equal(saved.create.value, './datasets');
  assert.equal(saved.update.value, './datasets');
});

test('gallery API and file-server resolve the same dataset root', async () => {
  const { load } = harness('./datasets');
  const paths = load('src/paths.ts');
  assert.equal(await load('src/server/settings.ts').getDatasetsRoot(), path.join(paths.TOOLKIT_ROOT, 'datasets'));
  assert.equal(load('cron/paths.ts').resolveDatasetsRoot('./datasets'), path.join(paths.TOOLKIT_ROOT, 'datasets'));
  const source = fs.readFileSync(path.join(root, 'ui/cron/fileServer.ts'), 'utf8');
  assert.match(source, /key === 'DATASETS_FOLDER'[\s\S]*?resolveDatasetsRoot/);
});
