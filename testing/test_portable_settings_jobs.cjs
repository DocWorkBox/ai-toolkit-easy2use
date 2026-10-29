const assert = require('node:assert/strict');
const { test, before, after, beforeEach } = require('node:test');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { createRequire } = require('node:module');

const root = path.resolve(process.env.AITK_TEST_ROOT || path.join(__dirname, '..'));
const ui = path.join(root, 'ui');
const req = createRequire(path.join(ui, 'package.json'));
const ts = req('typescript');
const { PrismaClient } = req('@prisma/client');
const scratch = fs.mkdtempSync(path.join(process.env.AITK_TEST_TMP || __dirname, '.lifecycle-test-'));
const db = new PrismaClient({ datasources: { db: { url: 'file:' + path.join(scratch, 'isolated.db').replaceAll('\\', '/') } } });
const quiet = { log() {}, warn() {}, error() {} };
const simProcess = { ...process, cwd: () => ui, env: { ...process.env, MODELS_PATH: '' } };

function loader(overrides = {}, globals = {}) {
  const cache = new Map();
  function load(relative, expose = '') {
    const filename = path.resolve(ui, relative);
    if (cache.has(filename)) return cache.get(filename);
    const source = ts.transpileModule(fs.readFileSync(filename, 'utf8') + expose, {
      compilerOptions: { module: ts.ModuleKind.CommonJS, esModuleInterop: true, target: ts.ScriptTarget.ES2022, jsx: ts.JsxEmit.ReactJSX },
    }).outputText;
    const mod = { exports: {} };
    const requireForTest = name => {
      if (Object.hasOwn(overrides, name)) return overrides[name];
      if (['@/server/prisma', './prisma', '../prisma'].includes(name)) return { __esModule: true, default: db };
      if (name === '@/utils/api') return { apiClient: {} };
      if (name === '@/server/settings') return { flushCache() {} };
      if (name === 'lucide-react') return { RefreshCw: () => null };
      let location;
      if (name.startsWith('@/')) location = path.join(ui, 'src', name.slice(2));
      if (name.startsWith('.')) location = path.resolve(path.dirname(filename), name);
      if (location) {
        for (const suffix of ['', '.ts', '.tsx']) {
          const candidate = location + suffix;
          if (fs.existsSync(candidate) && fs.statSync(candidate).isFile()) return load(path.relative(ui, candidate));
        }
        throw new Error('Missing local import ' + name);
      }
      return req(name);
    };
    vm.runInNewContext(source, {
      module: mod, exports: mod.exports, require: requireForTest, process: simProcess,
      console: quiet, Buffer, Request, Response, URL, setTimeout, clearTimeout, setInterval, clearInterval,
      ...globals,
    }, { filename });
    cache.set(filename, mod.exports);
    return mod.exports;
  }
  return load;
}

const bodyRequest = body => new Request('http://test/api/settings', {
  method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
});
const defaults = { HF_TOKEN: '', TRAINING_FOLDER: '', DATASETS_FOLDER: '', MODELS_PATH: '' };
const tick = () => new Promise(resolve => setTimeout(resolve, 10));
const params = { params: Promise.resolve({ jobID: 'job' }) };
const elements = node => {
  if (!node || typeof node !== 'object') return [];
  const children = node.props?.children;
  return [node, ...(Array.isArray(children) ? children.flat(Infinity) : [children]).flatMap(elements)];
};
function hooks() {
  let index = 0;
  const states = [], effects = [];
  return {
    states, effects, reset() { index = 0; },
    react: {
      ...req('react'),
      useState(initial) {
        const i = index++;
        if (!(i in states)) states[i] = initial;
        return [states[i], value => states[i] = typeof value === 'function' ? value(states[i]) : value];
      },
      useCallback: fn => fn,
      useRef(value) { const i = index++; if (!(i in states)) states[i] = { current: value }; return states[i]; },
      useEffect: fn => effects.push(fn),
    },
  };
}

before(async () => {
  await db.$executeRawUnsafe('CREATE TABLE Settings (id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT NOT NULL UNIQUE, value TEXT NOT NULL)');
  await db.$executeRawUnsafe('CREATE TABLE Queue (id INTEGER PRIMARY KEY AUTOINCREMENT, gpu_ids TEXT NOT NULL UNIQUE, is_running BOOLEAN NOT NULL DEFAULT false)');
  await db.$executeRawUnsafe(`CREATE TABLE Job (
    id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE, gpu_ids TEXT NOT NULL, job_config TEXT NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    status TEXT NOT NULL DEFAULT 'stopped', stop BOOLEAN NOT NULL DEFAULT false, return_to_queue BOOLEAN NOT NULL DEFAULT false,
    step INTEGER NOT NULL DEFAULT 0, total_steps INTEGER, info TEXT NOT NULL DEFAULT '', speed_string TEXT NOT NULL DEFAULT '',
    queue_position INTEGER NOT NULL DEFAULT 0, pid INTEGER, job_type TEXT NOT NULL DEFAULT 'train', job_ref TEXT,
    save_now BOOLEAN NOT NULL DEFAULT false, sample_now BOOLEAN NOT NULL DEFAULT false)`);
});
beforeEach(async () => { await db.settings.deleteMany(); await db.job.deleteMany(); await db.queue.deleteMany(); });
after(async () => { await db.$disconnect(); console.log('Isolated database:', scratch); });

test('Windows paths round-trip, including spaces, Unicode, UNC and blank defaults', async () => {
  const route = loader()('src/app/api/settings/route.ts');
  for (const value of [String.raw`F:\Training Data\datasets`, 'F:/Training Data/datasets', String.raw`F:\训练数据\datasets`, String.raw`\\nas\share\datasets`, '']) {
    const response = await route.POST(bodyRequest({ ...defaults, DATASETS_FOLDER: value, TRAINING_FOLDER: value }));
    assert.equal(response.status, 200);
    const saved = await (await route.GET()).json();
    assert.equal(saved.DATASETS_FOLDER, value || './datasets');
    assert.equal(saved.TRAINING_FOLDER, value || './output');
  }
});

test('partial settings payload cannot reset existing paths', async () => {
  const route = loader()('src/app/api/settings/route.ts');
  await route.POST(bodyRequest({ ...defaults, TRAINING_FOLDER: 'F:/custom-output' }));
  assert.equal((await route.POST(bodyRequest({ HF_TOKEN: '' }))).status, 400);
  assert.equal((await (await route.GET()).json()).TRAINING_FOLDER, 'F:/custom-output');
});

test('a failed settings write rolls back all four values and reports the database error', async () => {
  const route = loader()('src/app/api/settings/route.ts');
  await route.POST(bodyRequest({ ...defaults, TRAINING_FOLDER: 'F:/original-output' }));
  await db.$executeRawUnsafe(`CREATE TRIGGER reject_models BEFORE UPDATE ON Settings WHEN OLD.key = 'MODELS_PATH' BEGIN SELECT RAISE(ABORT, 'database is read-only'); END`);
  try {
    const res = await route.POST(bodyRequest({ ...defaults, HF_TOKEN: 'hf_private_test_secret', TRAINING_FOLDER: 'F:/new-output' }));
    assert.equal(res.status, 500);
    const error = (await res.json()).error;
    assert.match(error, /P2003/); // SQLite trigger aborts are mapped by Prisma to P2003.
    assert.ok(!error.includes('hf_private_test_secret'));
    assert.equal((await (await route.GET()).json()).TRAINING_FOLDER, 'F:/original-output');
    assert.equal((await db.settings.findUnique({ where: { key: 'HF_TOKEN' } })).value, '');
  } finally { await db.$executeRawUnsafe('DROP TRIGGER reject_models'); }
});

test('settings cannot edit or submit before loading, including after a load failure', async () => {
  for (const settingsError of [null, '读取设置失败: 数据库被锁定']) {
    const h = hooks();
    let posts = 0;
    const page = loader({
      react: h.react,
      '@/hooks/useSettings': { __esModule: true, default: () => ({ settings: defaults, setSettings() {}, isSettingsLoaded: false, settingsError, reloadSettings() {} }) },
      '@/components/layout': { TopBar: 'header', MainContent: 'main' },
      '@/utils/api': { apiClient: { post: async () => posts++ } },
    })('src/app/settings/page.tsx').default;
    const all = elements(page());
    assert.equal(all.find(n => n.type === 'button' && n.props.type === 'submit').props.disabled, true);
    assert.ok(all.some(n => n.type === 'fieldset' && n.props.disabled) || all.filter(n => n.type === 'input').every(n => n.props.disabled));
    await all.find(n => n.type === 'form').props.onSubmit({ preventDefault() {} });
    assert.equal(posts, 0);
    if (settingsError) assert.ok(all.some(n => n.props?.role === 'alert'));
  }
});

test('database errors report useful safe details rather than a query containing secrets', () => {
  const describe = loader()('src/server/settingsError.ts').settingsError;
  for (const [code, message, expected] of [
    ['P1008', 'Operation timed out', /超时/],
    ['P2010', 'attempt to write a readonly database', /只读/],
    ['P2021', 'no such table: Settings', /数据库结构/],
    ['P2010', 'SQLITE_FULL: database or disk is full', /空间不足/],
    ['P1010', 'unable to open database file', /无法打开/],
  ]) {
    const detail = describe({ code, message: `HF_TOKEN: hf_private_test_secret; ${message}` });
    assert.match(detail, expected); assert.ok(detail.includes(code));
    assert.ok(!detail.includes('hf_private_test_secret'));
  }
});

test('settings hook exposes HTTP errors and can reload without logging the HF token', async () => {
  const h = hooks(); let fail = true; const logs = [];
  const hook = loader({ react: h.react, '@/utils/api': { apiClient: { get: async () => {
    if (fail) throw { response: { status: 500, data: { error: '数据库只读 (SQLITE_READONLY)' } } };
    return { data: { ...defaults, HF_TOKEN: 'hf_private_test_secret', TRAINING_FOLDER: 'F:/my-output' } };
  } } } }, { console: { log: (...a) => logs.push(a), error: (...a) => logs.push(a) } })('src/hooks/useSettings.tsx').default;
  hook(); await h.effects[0](); await tick(); h.reset();
  let state = hook();
  assert.equal(state.isSettingsLoaded, false);
  assert.match(state.settingsError, /SQLITE_READONLY/);
  fail = false; await state.reloadSettings(); h.reset(); state = hook();
  assert.equal(state.settingsError, null);
  assert.equal(state.isSettingsLoaded, true);
  assert.equal(state.settings.TRAINING_FOLDER, 'F:/my-output');
  assert.ok(!JSON.stringify(logs).includes('hf_private_test_secret'));
});

test('settings save failure remains visible with the backend detail', async () => {
  const h = hooks(); const timeouts = [];
  const page = loader({
    react: h.react,
    '@/hooks/useSettings': { __esModule: true, default: () => ({ settings: defaults, setSettings() {}, isSettingsLoaded: true }) },
    '@/components/layout': { TopBar: 'header', MainContent: 'main' },
    '@/utils/api': { apiClient: { post: async () => { throw { response: { status: 500, data: { error: 'SQLITE_FULL: 磁盘已满' } } }; } } },
  }, { setTimeout: fn => timeouts.push(fn) })('src/app/settings/page.tsx').default;
  await elements(page()).find(n => n.type === 'form').props.onSubmit({ preventDefault() {} });
  await tick(); timeouts.forEach(fn => fn()); h.reset();
  const html = req('react-dom/server').renderToStaticMarkup(page());
  assert.match(html, /SQLITE_FULL/);
});

test('a delayed initial settings request cannot overwrite a newer load or user edits', async () => {
  const h = hooks(), pending = [];
  const hook = loader({ react: h.react, '@/utils/api': { apiClient: { get: () => new Promise(resolve => pending.push(resolve)) } } })('src/hooks/useSettings.tsx').default;
  let state = hook();
  const older = state.reloadSettings();
  const newer = state.reloadSettings();
  pending[1]({ data: { ...defaults, TRAINING_FOLDER: 'F:/saved' } }); await newer;
  h.reset(); state = hook(); state.setSettings({ ...state.settings, TRAINING_FOLDER: 'F:/editing' });
  pending[0]({ data: { ...defaults, TRAINING_FOLDER: './output' } }); await older;
  h.reset(); assert.equal(hook().settings.TRAINING_FOLDER, 'F:/editing');
});

async function jobRow(data = {}) {
  await db.queue.create({ data: { gpu_ids: '0', is_running: true } });
  return db.job.create({ data: { id: 'job', name: 'isolated', gpu_ids: '0', job_config: JSON.stringify({ config: { process: [{ train: { steps: 100 } }] } }), status: 'running', pid: 12345, ...data } });
}
function lifecycle(alive = new Set([12345])) {
  const intervals = [], kills = [], launched = [];
  const globals = {
    process: { ...simProcess, platform: 'win32', kill(pid, signal) {
      if (alive.has(pid)) return true;
      const e = new Error('exited'); e.code = 'ESRCH'; throw e;
    } },
    setInterval: fn => { const t = { fn, unref() {} }; intervals.push(t); return t; }, clearInterval() {},
  };
  const load = loader({
    child_process: { exec: (cmd, opts, cb) => { kills.push(cmd); cb(null, '', ''); }, spawn() { throw new Error('No training process may be launched'); } },
    './startJob': { __esModule: true, default: async id => launched.push(id) },
  }, globals);
  return { load, intervals, kills, launched, alive, globals };
}

test('stop keeps an alive process stopping and rejects premature resume', async () => {
  await jobRow(); const h = lifecycle();
  const res = await h.load('src/app/api/jobs/[jobID]/stop/route.ts').GET(null, params);
  const job = await db.job.findUnique({ where: { id: 'job' } });
  assert.equal(job.status, 'stopping');
  assert.equal((await res.json()).status, 'stopping');
  assert.equal(h.load('src/utils/jobs.ts').getAvaliableJobActions(job).canStart, false);
  const start = await h.load('src/app/api/jobs/[jobID]/start/route.ts').GET(null, params);
  assert.equal(start.status, 409);
  assert.equal((await db.job.findUnique({ where: { id: 'job' } })).stop, true);
});

test('the UI also waits for the PID after Python reports stopped early', async () => {
  const job = await jobRow({ status: 'stopped', stop: true });
  assert.equal(lifecycle().load('src/utils/jobs.ts').getAvaliableJobActions(job).canStart, false);
});

test('queue cannot launch another job while the stopping process is still alive', async () => {
  await jobRow({ status: 'stopped', stop: true });
  await db.job.create({ data: { id: 'next', name: 'next', gpu_ids: '0', job_config: '{}', status: 'queued' } });
  const h = lifecycle(); await h.load('cron/actions/processQueue.ts').default();
  assert.equal(h.launched.length, 0);
});

test('dead PID is cleared, stopped job resumes and restarts its paused queue', async () => {
  await jobRow({ status: 'stopped', stop: true });
  const h = lifecycle(new Set());
  await h.load('cron/actions/processQueue.ts').default();
  assert.equal((await db.job.findUnique({ where: { id: 'job' } })).pid, null);
  await db.queue.updateMany({ data: { is_running: false } });
  assert.equal((await h.load('src/app/api/jobs/[jobID]/start/route.ts').GET(null, params)).status, 200);
  assert.equal((await db.queue.findUnique({ where: { gpu_ids: '0' } })).is_running, true);
  await h.load('cron/actions/processQueue.ts').default();
  assert.deepEqual(h.launched, ['job']);
});

test('an old detached-process watcher cannot mark a new run as failed', async () => {
  await jobRow({ pid: 56789 }); const h = lifecycle(new Set([56789]));
  const watcher = h.load('cron/actions/startJob.ts', '\nexport const testWatch = watchDetachedJob;');
  watcher.testWatch(12345, 'job', path.join(scratch, 'missing.log'));
  await h.intervals.at(-1).fn(); await tick();
  const job = await db.job.findUnique({ where: { id: 'job' } });
  assert.equal(job.status, 'running'); assert.equal(job.pid, 56789);
});

test('exited stopping process becomes stopped rather than an unexpected error', async () => {
  await jobRow({ status: 'stopping', stop: true }); const h = lifecycle(new Set());
  await h.load('cron/actions/processQueue.ts').default();
  const job = await db.job.findUnique({ where: { id: 'job' } });
  assert.equal(job.status, 'stopped'); assert.equal(job.pid, null);
});

test('return-to-queue waits for exit, while normal completion is retained and crashes are errors', async () => {
  for (const [status, stop, return_to_queue, expected] of [
    ['running', false, true, 'queued'], ['running', false, false, 'error'], ['completed', false, false, 'completed'],
  ]) {
    await db.job.deleteMany(); await db.queue.deleteMany();
    await jobRow({ status, stop, return_to_queue });
    await db.queue.updateMany({ data: { is_running: false } });
    const h = lifecycle(new Set()); await h.load('cron/actions/processQueue.ts').default();
    const job = await db.job.findUnique({ where: { id: 'job' } });
    assert.equal(job.status, expected); assert.equal(job.pid, null);
  }
});

test('resume queue failure does not leave the job permanently queued with its stop flag cleared', async () => {
  await jobRow({ status: 'stopped', stop: true, pid: null });
  await db.queue.updateMany({ data: { is_running: false } });
  await db.$executeRawUnsafe(`CREATE TRIGGER reject_queue BEFORE UPDATE ON Queue BEGIN SELECT RAISE(ABORT, 'isolated failure'); END`);
  try {
    try { await lifecycle(new Set()).load('src/app/api/jobs/[jobID]/start/route.ts').GET(null, params); } catch {}
    const job = await db.job.findUnique({ where: { id: 'job' } });
    assert.equal(job.status, 'stopped'); assert.equal(job.stop, true);
  } finally { await db.$executeRawUnsafe('DROP TRIGGER reject_queue'); }
});

test('an idle queue check cannot undo a resume that just queued a job', async () => {
  await jobRow({ status: 'stopped', stop: true, pid: null });
  const racingDb = { ...db, job: { ...db.job, findFirst: async args => {
    const result = await db.job.findFirst(args);
    if (args.where.status === 'queued' && !result) {
      await db.job.update({ where: { id: 'job' }, data: { status: 'queued', stop: false } });
    }
    return result;
  } }, $transaction: db.$transaction.bind(db) };
  await loader({ '../prisma': { __esModule: true, default: racingDb } })('cron/actions/processQueue.ts').default();
  assert.equal((await db.queue.findUnique({ where: { gpu_ids: '0' } })).is_running, true);
});

test('model loading errors render module, file and reason, with a retry button', async () => {
  const file = path.join(ui, 'src/components/ModelLoadErrors.tsx');
  assert.ok(fs.existsSync(file), 'model loader errors need a visible component');
  const component = loader()('src/components/ModelLoadErrors.tsx').default;
  let retries = 0;
  const tree = component({ errors: ['extensions_built_in/diffusion_models (C:/portable/ui.tsx): cannot import xyz'], onRetry: () => retries++ });
  const html = req('react-dom/server').renderToStaticMarkup(tree);
  assert.match(html, /diffusion_models/); assert.match(html, /cannot import xyz/); assert.match(html, /role="alert"/);
  await elements(tree).find(n => n.type === 'button').props.onClick(); assert.equal(retries, 1);
  for (const relative of ['src/app/jobs/new/SimpleJob.tsx', 'src/app/generate/page.tsx']) {
    assert.match(fs.readFileSync(path.join(ui, relative), 'utf8'), /<ModelLoadErrors/);
  }
});

test('model loader retains healthy modules and exposes file-qualified failures and retry recovery', async () => {
  let state, broken = true;
  const good = { id: 'extensions_built_in/flex2', file: 'C:/portable/flex2/ui.tsx', code: 'exports.AI_TOOLKIT_UI_MODELS = [{name:"flex2",label:"Flex.2",group:"image"}]' };
  const bad = { id: 'extensions_built_in/diffusion_models', file: 'C:/portable/diffusion_models/ui.tsx', code: 'throw new Error("cannot import xyz")' };
  const load = loader({
    'react-global-hooks': { createGlobalState(initial) { state = initial; return { get: () => state, set: v => state = v, use: () => [state] }; } },
    '@/components/formInputs': {},
    '@/utils/api': { apiClient: { get: async () => ({ data: { modules: [good, { ...bad, code: broken ? bad.code : 'exports.AI_TOOLKIT_UI_MODELS = [{name:"moss_image",label:"MOSS",group:"image"}]' }], errors: [] } }) } },
  });
  const hook = load('src/extensions/modelArchs.ts');
  await hook.loadModelArchs();
  assert.equal(state.archs.length, 1);
  assert.match(state.errors[0], /C:\/portable\/diffusion_models\/ui.tsx/);
  assert.match(state.errors[0], /cannot import xyz/);
  broken = false; await hook.loadModelArchs();
  assert.equal(state.archs.length, 2); assert.equal(state.errors.length, 0);
});

test('real Windows child process must exit before a paused job can resume', { skip: process.platform !== 'win32', timeout: 15000 }, async () => {
  const { spawn } = require('node:child_process');
  const { once } = require('node:events');
  const child = spawn(process.execPath, ['-e', 'process.on("message", m => { if (m === "exit") process.exit(0); }); process.send("ready");'], { windowsHide: true, stdio: ['ignore', 'ignore', 'pipe', 'ipc'] });
  try {
    await once(child, 'message');
    await jobRow({ pid: child.pid });
    const load = loader({}, { setInterval: () => ({ unref() {} }), clearInterval() {} });
    const stop = load('src/app/api/jobs/[jobID]/stop/route.ts');
    const start = load('src/app/api/jobs/[jobID]/start/route.ts');
    assert.equal((await stop.GET(null, params)).status, 200);
    assert.equal((await start.GET(null, params)).status, 409);
    await db.job.update({ where: { id: 'job' }, data: { status: 'stopped' } });
    assert.equal((await start.GET(null, params)).status, 409);
    const exit = once(child, 'exit'); child.send('exit'); await exit;
    assert.equal((await start.GET(null, params)).status, 200);
    const resumed = await db.job.findUnique({ where: { id: 'job' } });
    assert.equal(resumed.status, 'queued'); assert.equal(resumed.pid, null); assert.equal(resumed.stop, false);
    assert.equal((await db.queue.findUnique({ where: { gpu_ids: '0' } })).is_running, true);
  } finally { if (child.exitCode === null) { const exit = once(child, 'exit'); child.kill(); await exit; } }
});
