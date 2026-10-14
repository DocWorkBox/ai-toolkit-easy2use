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

function harness(value, trainingValue, job) {
  const rows = value === undefined ? [] : [{ key: 'DATASETS_FOLDER', value }];
  if (trainingValue !== undefined) rows.push({ key: 'TRAINING_FOLDER', value: trainingValue });
  const writes = [];
  const prisma = { settings: {
    findMany: async () => rows,
    findFirst: async ({ where }) => rows.find(row => row.key === where.key),
    upsert: async args => writes.push(args),
  }, job: { findUnique: async () => job } };
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
    vm.runInNewContext(source, { module, exports: module.exports, require, process, console, Buffer }, { filename });
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

test('training output defaults are relative in settings and the job template', async () => {
  const { load } = harness();
  assert.equal(load('src/paths.ts').defaultTrainFolder, './output');
  assert.equal(load('cron/paths.ts').defaultTrainFolder, './output');
  assert.equal((await load('src/app/api/settings/route.ts').GET()).TRAINING_FOLDER, './output');
  const template = fs.readFileSync(path.join(root, 'ui/src/app/jobs/new/jobConfig.ts'), 'utf8');
  assert.match(template, /training_folder: '\.\/output'/);
});

test('training output follows the toolkit root across drives and spaces', () => {
  const helper = harness().load('cron/trainingPaths.ts');
  for (const root of ['C:\\portable-build\\ai-toolkit-easy2use-portable', 'F:\\Training Tools\\portable']) {
    for (const value of [undefined, '', '  ', './output', 'output', '.\\output\\']) {
      assert.equal(helper.normalizeTrainingSetting(value, root, path.win32), './output');
      assert.equal(helper.resolveTrainingRoot(value, root, path.win32), path.win32.join(root, 'output'));
    }
    assert.equal(helper.resolveTrainingRoot('./runs', root, path.win32), path.win32.join(root, 'runs'));
  }
});

test('external and shared output directories are not rewritten', () => {
  const helper = harness().load('cron/trainingPaths.ts');
  for (const value of ['E:\\training-output', '\\\\nas\\share\\output']) {
    assert.equal(helper.normalizeTrainingSetting(value, 'F:\\portable', path.win32), value);
    assert.equal(helper.resolveTrainingRoot(value, 'F:\\portable', path.win32), value);
  }
  assert.equal(helper.normalizeTrainingSetting('c:\\Portable\\output\\', 'C:\\portable', path.win32), './output');
  assert.equal(helper.resolveTrainingRoot('./output', '/opt/portable', path.posix), '/opt/portable/output');
});

test('training settings GET and POST normalize only the local default output', async () => {
  const defaults = harness().load('src/paths.ts');
  for (const value of ['', '   ', './output', path.join(defaults.TOOLKIT_ROOT, 'output')]) {
    const { load, writes } = harness(undefined, value);
    const route = load('src/app/api/settings/route.ts');
    assert.equal((await route.GET()).TRAINING_FOLDER, './output');
    await route.POST({ json: async () => ({ TRAINING_FOLDER: value, DATASETS_FOLDER: './datasets', HF_TOKEN: '', MODELS_PATH: '' }) });
    const saved = writes.find(row => row.where.key === 'TRAINING_FOLDER');
    assert.equal(saved.create.value, './output');
    assert.equal(saved.update.value, './output');
  }
  const { load, writes } = harness(undefined, 'E:\\training-output');
  const route = load('src/app/api/settings/route.ts');
  assert.equal((await route.GET()).TRAINING_FOLDER, 'E:\\training-output');
  await route.POST({ json: async () => ({ TRAINING_FOLDER: 'E:\\training-output', DATASETS_FOLDER: './datasets', HF_TOKEN: '', MODELS_PATH: '' }) });
  assert.equal(writes.find(row => row.where.key === 'TRAINING_FOLDER').update.value, 'E:\\training-output');
});

test('job APIs, training worker and file server agree on the absolute output root', async () => {
  for (const value of [undefined, './output', './runs', path.join(root, 'external-output')]) {
    const { load } = harness(undefined, value);
    const paths = load('src/paths.ts');
    const expected = path.resolve(paths.TOOLKIT_ROOT, value || './output');
    assert.equal(await load('src/server/settings.ts').getTrainingFolder(), expected);
    assert.equal(await load('cron/paths.ts').getTrainingFolder(), expected);
    assert.equal(load('cron/paths.ts').resolveTrainingRoot(value), expected);
  }
  const source = fs.readFileSync(path.join(root, 'ui/cron/fileServer.ts'), 'utf8');
  assert.match(source, /key === 'TRAINING_FOLDER'[\s\S]*?resolveTrainingRoot/);
});

test('job routes read logs, checkpoints and samples under a root-relative output', async () => {
  const fixture = fs.mkdtempSync(path.join(root, '.output-path-test-'));
  const job = { id: 'path-test', name: 'test-job' };
  const jobFolder = path.join(fixture, job.name);
  const samplesFolder = path.join(jobFolder, 'samples');
  fs.mkdirSync(samplesFolder, { recursive: true });
  const files = [path.join(jobFolder, 'log.txt'), path.join(jobFolder, 'test.safetensors'), path.join(samplesFolder, 'sample.jpg')];
  try {
    for (const file of files) fs.writeFileSync(file, 'path test fixture', 'utf8');
    const { load } = harness(undefined, './' + path.basename(fixture), job);
    const params = { params: { jobID: job.id } };
    const request = { nextUrl: new URL('http://localhost/api/jobs/path-test/log') };
    const log = await load('src/app/api/jobs/[jobID]/log/route.ts').GET(request, params);
    assert.equal(log.log, 'path test fixture');
    const checkpoints = await load('src/app/api/jobs/[jobID]/files/route.ts').GET(request, params);
    assert.equal(checkpoints.files.length, 1);
    assert.equal(checkpoints.files[0].path, files[1]);
    const samples = await load('src/app/api/jobs/[jobID]/samples/route.ts').GET(request, params);
    assert.equal(samples.samples.length, 1);
    assert.equal(samples.samples[0], files[2]);
  } finally {
    for (const file of files) if (fs.existsSync(file)) fs.unlinkSync(file);
    fs.rmdirSync(samplesFolder);
    fs.rmdirSync(jobFolder);
    fs.rmdirSync(fixture);
  }
});
