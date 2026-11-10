import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const require = createRequire(resolve(root, 'ui/package.json'));
const ts = require('typescript');
const source = path => readFileSync(resolve(root, path), 'utf8');

function modelDefaults(path) {
  const file = ts.createSourceFile(path, source(path), ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
  const models = new Map();
  function visit(node) {
    if (ts.isObjectLiteralExpression(node)) {
      const properties = new Map(node.properties.filter(ts.isPropertyAssignment).map(p => [
        ts.isStringLiteral(p.name) ? p.name.text : p.name.getText(file), p.initializer,
      ]));
      const name = properties.get('name');
      const defaults = properties.get('defaults');
      if (name && ts.isStringLiteral(name) && defaults && ts.isObjectLiteralExpression(defaults)) {
        models.set(name.text, new Map(defaults.properties.filter(ts.isPropertyAssignment).map(p => [
          ts.isStringLiteral(p.name) ? p.name.text : p.name.getText(file),
          ts.isArrayLiteralExpression(p.initializer)
            ? p.initializer.elements[0]?.text ?? p.initializer.elements[0]?.getText(file)
            : undefined,
        ])));
      }
    }
    ts.forEachChild(node, visit);
  }
  visit(file);
  return models;
}

const diffusion = modelDefaults('extensions_built_in/diffusion_models/ui.tsx');
const oldDefaults = {
  flux: '/chenyudata/Other/flux.1-dev',
  flux_kontext: '/chenyudata/Other/FLUX.1-Kontext-dev',
  flex1: 'ostris/Flex.1-alpha',
  chroma: 'lodestones/Chroma1-Base',
  zeta_chroma: 'lodestones/Zeta-Chroma/zeta-chroma-base-x0-pixel-dino-distance.safetensors',
  'wan21:1b': '/chenyudata/Other/Wan2.1-T2V-1.3B-Diffusers',
  'wan21_i2v:14b480p': '/chenyudata/Other/Wan2.1-I2V-14B-480P-Diffusers',
  'wan21_i2v:14b': '/chenyudata/Other/Wan2.1-I2V-14B-720P-Diffusers',
  'wan21:14b': '/chenyudata/Other/Wan2.1-T2V-14B-Diffusers',
  'wan22_14b:t2v': '/chenyudata/Other/Wan2.2-T2V-A14B-Diffusers',
  wan22_14b_i2v: '/chenyudata/Other/Wan2.2-I2V-A14B-Diffusers',
  wan22_5b: '/chenyudata/Other/Wan2.2-TI2V-5B-Diffusers',
  lumina2: 'Alpha-VLLM/Lumina-Image-2.0',
  qwen_image: '/chenyudata/Other/Qwen-Image',
  'qwen_image:2512': '/chenyudata/Other/Qwen-Image-2512',
  qwen_image_edit: '/chenyudata/Other/Qwen-Image-Edit',
  qwen_image_edit_plus: '/chenyudata/Other/Qwen-Image-Edit-2509',
  'qwen_image_edit_plus:2511': '/chenyudata/Other/Qwen-Image-Edit-2511',
  'qwen_image_edit_plus:firered': '/chenyudata/Other/FireRed-Image-Edit-1.1',
  hidream: 'HiDream-ai/HiDream-I1-Full',
  hidream_e1: 'HiDream-ai/HiDream-E1-1',
  sdxl: 'stabilityai/stable-diffusion-xl-base-1.0',
  sd15: 'stable-diffusion-v1-5/stable-diffusion-v1-5',
  omnigen2: 'OmniGen2/OmniGen2',
  flux2: '/chenyudata/Other/FLUX.2-dev',
  'zimage:turbo': '/chenyudata/Other/Z-Image-Turbo',
  zimage: '/chenyudata/Other/Z-Image',
  'zimage:deturbo': '/chenyudata/Other/Z-Image-De-Turbo',
  ltx2: '/chenyudata/Other/LTX-2',
  'ltx2.3': '/chenyudata/ComfyUI/models/checkpoints/ltx/ltx-2.3-22b-dev.safetensors',
  flux2_klein_4b: '/chenyudata/Other/FLUX.2-klein-base-4B',
  flux2_klein_9b: '/chenyudata/Other/FLUX.2-klein-base-9B',
};
const discoveredDefaults = {
  anima: '/chenyudata/Other/Anima-Base-v1.0-Diffusers',
  qwen_image_2: '/chenyudata/Other/Comfy-Org/Qwen-Image-2.1',
  'qwen_image_2:turbo': '/chenyudata/Other/Comfy-Org/Qwen-Image-2.1',
  minimax_h3: '/chenyudata/Other/Comfy-Org/MiniMax-H3',
  minimax_h3_ref2va: '/chenyudata/Other/Comfy-Org/MiniMax-H3',
  'ltx2.5': '/chenyudata/Other/LTX-2.5',
  flux3: '/chenyudata/Other/flux-3-image-klein-base',
  moss_image: 'OpenMOSS-Team/MOSS-Image-SFT',
  'moss_image:base': 'OpenMOSS-Team/MOSS-Image-Base',
  'moss_image:edit': 'OpenMOSS-Team/MOSS-Image-SFT',
  'moss_image:base_edit': 'OpenMOSS-Team/MOSS-Image-Base',
  fibo2: 'Comfy-Org/Fibo-2.0',
  ernie_image: '/chenyudata/Other/ERNIE-Image',
  ideogram4: '/chenyudata/Other/ideogram-4-fp8',
  krea2: '/chenyudata/Other/Krea-2-Raw',
  'krea2:turbo': '/chenyudata/Other/Krea-2-Turbo',
  'krea2:o_edit': '/chenyudata/Other/Krea-2-Raw',
  'krea2:o_edit_turbo': '/chenyudata/Other/Krea-2-Turbo',
  boogu_image: '/chenyudata/Other/Boogu-Image-0.1-Base',
  boogu_image_edit: '/chenyudata/Other/Boogu-Image-0.1-Edit',
};
const missingDefaults = {
  p_h3_image: 'PrunaAI/P-H3-Image-Comfy',
  minimax_h3_omni: 'FastVideo/FastH3-Omni-Comfy',
  minimax_h3_vsa: 'FastVideo/FastVideo-FastH3-Comfy',
  ming_image: 'Comfy-Org/Ming-Image',
  nucleus_image: 'NucleusAI/Nucleus-Image',
  hidream_o1: 'HiDream-ai/HiDream-O1-Image',
  zimage_l2p: 'zhen-nan/L2P/model-1k-merge.safetensors',
  prx_pixel: 'Photoroom/prxpixel-t2i',
  iris3b: 'speridlabs/iris-3b',
  mageflow: 'microsoft/Mage-Flow-Base',
  mageflow_edit: 'microsoft/Mage-Flow-Edit-Base',
};

for (const [name, expected] of Object.entries({ ...oldDefaults, ...discoveredDefaults, ...missingDefaults })) {
  test(`${name} uses the verified chengyu source or official fallback`, () => {
    assert.equal(diffusion.get(name)?.get('config.process[0].model.name_or_path'), expected);
  });
}

test('current plugin architecture and auxiliary repo defaults are retained', () => {
  assert.equal(diffusion.size, 63);
  assert.equal(diffusion.has('flex2'), false);
  for (const name of ['zeta_chroma', 'zimage:deturbo']) {
    assert.equal(diffusion.get(name).get('config.process[0].model.extras_name_or_path'), 'Tongyi-MAI/Z-Image-Turbo');
  }
  assert.equal(diffusion.get('zimage:turbo').get('config.process[0].model.assistant_lora_path'),
    'ostris/zimage_turbo_training_adapter/zimage_turbo_training_adapter_v2.safetensors');
  assert.equal(diffusion.get('zimage_l2p').get('config.process[0].model.extras_name_or_path'),
    '/chenyudata/Other/Z-Image-Turbo');
});

test('captioning uses available Qwen3-VL weights without substituting other Qwen versions', () => {
  const captions = modelDefaults('ui/src/helpers/captionOptions.ts');
  assert.equal(captions.get('Qwen3VLCaptioner').get('config.process[0].caption.model_name_or_path'),
    '/chenyudata/Other/Qwen3-VL-8B-Instruct');
  assert.equal(captions.get('Qwen38Captioner').get('config.process[0].caption.model_name_or_path'),
    'Comfy-Org/Qwen3.8-27B/text_encoders/qwen3.8_27b_w4a8.safetensors');
});

test('FastH3 variants reuse the verified local H3 text encoder and VAEs', () => {
  for (const arch of ['minimax_h3_vsa', 'minimax_h3_omni']) {
    const kwargs = diffusion.get(arch).get('config.process[0].model.model_kwargs') ?? '';
    for (const file of [
      'text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors',
      'vae/minimax_h3_video_vae_fp16.safetensors',
      'vae/minimax_h3_audio_vae_fp32.safetensors',
    ]) {
      assert.ok(kwargs.includes(`/chenyudata/Other/Comfy-Org/MiniMax-H3/${file}`), `${arch}: ${file}`);
    }
  }
});
