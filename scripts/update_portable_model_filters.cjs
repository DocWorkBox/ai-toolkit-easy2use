// Regenerate after changing the bundled trainer/caption presets. Only read
// literal metadata; never execute an extension to build the launcher catalog.
const fs = require('node:fs');
const path = require('node:path');
const ts = require('../ui/node_modules/typescript');
const root = path.resolve(__dirname, '..');
const file = path.join(root, 'portable_models.json');
const catalog = JSON.parse(fs.readFileSync(file, 'utf8'));
const rows = catalog.models;
const ids = new Set(rows.map(m => m.id));
const family = name => rows.filter(m => m.family === name).map(m => m.id);
const prop = (o, key) => o?.properties?.find(p => p.name?.text === key)?.initializer;
const field = (o, key) => prop(o, key)?.text;
const byPath = p => rows.filter(m => m.path === p).map(m => m.id);
function options(filename, symbol) {
  const tree = ts.createSourceFile(filename, fs.readFileSync(filename, 'utf8'), ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
  const declaration = tree.statements.flatMap(s => [...(s.declarationList?.declarations ?? [])]).find(d => d.name.text === symbol);
  if (!declaration) return [];
  let array = declaration.initializer;
  if (ts.isAsExpression(array)) array = array.expression;
  if (ts.isCallExpression(array) && array.expression.name?.text === 'sort') array = array.expression.expression;
  if (!ts.isArrayLiteralExpression(array)) throw Error('Unsupported preset definition: ' + filename);
  return array.elements;
}

const h3 = ['minimax-h3-text-encoder','minimax-h3-video-vae','minimax-h3-audio-vae','minimax-h3-tokenizer','minimax-h3-processor','minimax-h3-text-encoder-config'];
const qi = ['qwen-image-21-config','qwen-image-21-text-encoder','qwen-image-21-vae'];
const krea = ['qwen3-vl-4b','qwen-image-vae'];
function moss(variant, edit = false) {
  return ['moss-image-' + variant, 'moss-image-' + variant + '-tokenizer', 'moss-image-text-encoder','moss-image-vae','moss-image-text-config', ...(edit ? ['moss-image-edit-processor'] : [])];
}
// Dependencies resolved by Python loaders rather than explicit paths in the form.
const extra = {
  p_h3_image: family('P-H3-Image'), ming_image: family('Ming-Image'), fibo2: family('Fibo 2.0'), flux3: family('FLUX.3 Image Klein'),
  'ltx2.5': family('LTX-2.5'), yue2: family('YuE2'), 'ltx2.3': ['ltx2-text-encoder'],
  qwen_image_2: ['qwen-image-21-transformer', ...qi], 'qwen_image_2:turbo': ['qwen-image-21-turbo-transformer', ...qi],
  minimax_h3: ['minimax-h3-fl2va', ...h3], minimax_h3_ref2va: ['minimax-h3-ref2va', ...h3],
  minimax_h3_omni: ['fasth3-omni-transformer', ...h3], minimax_h3_vsa: ['fasth3-v2-transformer', ...h3],
  moss_image: moss('sft'), 'moss_image:base': moss('base'), 'moss_image:edit': moss('sft', true), 'moss_image:base_edit': moss('base', true),
  krea2: krea, 'krea2:turbo': krea, 'krea2:o_edit': krea, 'krea2:o_edit_turbo': krea,
  flux2: ['flux2-vae','flux2-mistral'], flux2_klein_4b: ['flux2-vae','flux2-klein-4b-text-encoder'], flux2_klein_9b: ['flux2-vae','flux2-klein-9b-text-encoder'],
  hidream: ['hidream-llama'], hidream_e1: ['hidream-llama'], iris3b: ['iris3b-text-encoder'], ideogram4: ['qwen3-vl-8b'],
  ace_step_15: ['qwen3-embedding-06b'], ace_step_15_xl: ['qwen3-embedding-06b'],
  qwen25_omni: ['qwen25-omni-7b-metadata'], qwen38: ['qwen38-assets'],
};
const groups = { image:'图像', video:'视频', instruction:'指令编辑', experimental:'实验', audio:'音频', llm:'语言模型' };
const filters = [];
function add(id, label, group, models, remote = false) {
  models = [...new Set(models)];
  if (!label || !group || (!models.length && !remote) || models.some(x => !ids.has(x)) || filters.some(f => f.id === id))
    throw Error('Invalid filter: ' + id + ' ' + models);
  filters.push({id, label, group, models, ...(remote ? {remote:true} : {})});
}
const builtins = path.join(root, 'extensions_built_in');
for (const dir of fs.readdirSync(builtins).sort()) {
  const ui = path.join(builtins, dir, 'ui.tsx');
  if (!fs.existsSync(ui)) continue;
  for (const o of options(ui, 'AI_TOOLKIT_UI_MODELS')) {
    const id = field(o,'name');
    const defaults = (prop(o,'defaults')?.properties ?? []).flatMap(p => byPath(p.initializer?.elements?.[0]?.text));
    const wan = id.startsWith('wan') ? ['wan-umt5', ...(id.startsWith('wan22_14b') ? ['wan-vae'] : [])] : [];
    add(id, field(o,'label'), groups[field(o,'group')], [...defaults, ...wan, ...(extra[id] ?? [])]);
  }
}
for (const o of options(path.join(root,'ui/src/helpers/captionOptions.ts'), 'captionerTypes')) {
  const id = 'caption/' + field(o,'name'), title = field(o,'label');
  const def = key => prop(prop(o,'defaults'), 'config.process[0].caption.' + key)?.elements?.[0]?.text;
  const choices = prop(o,'name_or_path_options')?.elements;
  const variants = choices?.map(c => ({value:field(c,'value'), label:field(c,'label')})) ?? [{value:def('model_name_or_path'), label:title}];
  for (const v of variants) {
    const remote = ['caption/RemoteAPICaptioner', 'caption/Ideogram4APICaptioner'].includes(id);
    let matches = byPath(v.value);
    if (v.value === './models/Qwen3-VL-4B-Instruct') matches = ['qwen3-vl-4b'];
    if (!matches.length && !remote) throw Error('Missing weight entry: ' + id + ' ' + v.value);
    const deps = {
      'qwen38-w4a8':['qwen38-assets'], 'qwen25-omni-checkpoint':['qwen25-omni-7b-metadata'],
      'qwen3-omni-instruct':['qwen3-omni-instruct-metadata'], 'qwen3-omni-thinking':['qwen3-omni-thinking-metadata'],
      'qwen3-omni-thinking-abliterated':['qwen3-omni-thinking-metadata']
    };
    const multi = variants.length > 1;
    add(id + (multi ? '/' + path.posix.basename(v.value) : ''), title + (multi ? ' · ' + v.label.replace(/^\.\/models\//, '') : ''), '打标模型',
      [...matches, ...byPath(def('model_name_or_path2')), ...matches.flatMap(m => deps[m] ?? [])], remote);
  }
}

filters.sort((a,b) => a.label.localeCompare(b.label, 'en'));
if (process.argv.includes('--check')) {
  if (JSON.stringify(filters) !== JSON.stringify(catalog.filters)) throw Error('Run node scripts/update_portable_model_filters.cjs');
} else {
  catalog.filters = filters;
  // Keep the catalog's one-record-per-line format for readable diffs.
  const parts = Object.entries(catalog).map(([key,value]) => '  ' + JSON.stringify(key) + ': ' +
    (Array.isArray(value) ? '[\n' + value.map(v => '    ' + JSON.stringify(v)).join(',\n') + '\n  ]' : JSON.stringify(value)));
  fs.writeFileSync(file, '{\n' + parts.join(',\n') + '\n}\n');
}
console.log(filters.length + ' portable model filters');
