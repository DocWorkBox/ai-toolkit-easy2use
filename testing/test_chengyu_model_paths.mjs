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
  wan22_5b: '/chenyudata/ComfyUI/models/diffusion_models/Wan_2.2_ComfyUI_Repackaged/split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors',
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
  p_h3_image: '/poddata/ai-toolkit-models',
  minimax_h3_omni: '/poddata/ai-toolkit-models',
  minimax_h3_vsa: '/poddata/ai-toolkit-models',
  ming_image: 'Comfy-Org/Ming-Image',
  nucleus_image: 'NucleusAI/Nucleus-Image',
  hidream_o1: 'HiDream-ai/HiDream-O1-Image',
  zimage_l2p: '/poddata/ai-toolkit-models/diffusion_models/model-1k-merge.safetensors',
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
    '/poddata/ai-toolkit-models/loras/training_adapters/zimage_turbo_training_adapter_v2.safetensors');
  assert.equal(diffusion.get('zimage_l2p').get('config.process[0].model.extras_name_or_path'),
    '/chenyudata/Other/Z-Image-Turbo');
});

test('Wan2.2 5B reuses its matching VAE and the existing UMT5 checkpoint', () => {
  const defaults = diffusion.get('wan22_5b');
  assert.equal(defaults.get('config.process[0].model.extras_name_or_path'),
    'Wan-AI/Wan2.2-TI2V-5B-Diffusers');
  assert.equal(defaults.get('config.process[0].model.vae_path'),
    '/chenyudata/ComfyUI/models/vae/wan2.2_vae.safetensors');
  assert.equal(defaults.get('config.process[0].model.te_name_or_path'),
    '/chenyudata/Other/Wan2.1-T2V-1.3B-Diffusers');
});

test('captioning uses available Qwen3-VL weights without substituting other Qwen versions', () => {
  const captions = modelDefaults('ui/src/helpers/captionOptions.ts');
  assert.equal(captions.get('Qwen3VLCaptioner').get('config.process[0].caption.model_name_or_path'),
    '/chenyudata/Other/Qwen3-VL-8B-Instruct');
  assert.equal(captions.get('Qwen38Captioner').get('config.process[0].caption.model_name_or_path'),
    '/poddata/ai-toolkit-models/text_encoders/qwen3.8_27b_w4a8.safetensors');
});

test('downloaded audio and language models are preconfigured at their final paths', () => {
  const audio = modelDefaults('extensions_built_in/audio_models/ui.tsx');
  const llm = modelDefaults('extensions_built_in/llm_models/ui.tsx');
  for (const [arch, filename] of Object.entries({
    ace_step_15: 'ace_step_1.5_base_aio.safetensors',
    ace_step_15_xl: 'ace_step_1.5_xl_base_aio.safetensors',
    yue2: 'yue2_3b_int8_convrot.safetensors',
  })) {
    assert.equal(audio.get(arch).get('config.process[0].model.name_or_path'),
      `/poddata/ai-toolkit-models/checkpoints/${filename}`);
  }
  assert.equal(llm.get('qwen38').get('config.process[0].model.name_or_path'),
    '/poddata/ai-toolkit-models/text_encoders/qwen3.8_27b_w4a8.safetensors');
});

test('captioner defaults point to the downloaded HF snapshots and Huihui checkpoint', () => {
  const captions = modelDefaults('ui/src/helpers/captionOptions.ts');
  for (const [name, relative] of Object.entries({
    AceStepCaptioner: 'huggingface/ACE-Step/acestep-transcriber',
    MossMusicCaptioner: 'huggingface/OpenMOSS-Team/MOSS-Music-8B-Instruct',
    Qwen3OmniCaptioner: 'text_encoders/huihui_qwen3_omni_30b_a3b_thinking_abliterated_convrot8.safetensors',
  })) {
    assert.equal(captions.get(name).get('config.process[0].caption.model_name_or_path'),
      `/poddata/ai-toolkit-models/${relative}`);
  }
  assert.equal(captions.get('AceStepCaptioner').get('config.process[0].caption.model_name_or_path2'),
    '/poddata/ai-toolkit-models/huggingface/ACE-Step/acestep-captioner');
});

test('training adapters and option callbacks retain the final local filenames', () => {
  const adapters = {
    'qwen_image_2:turbo': 'qwen_image_2_turbo_training_adapter_v1.safetensors',
    ming_image: 'ming_image_01_design_training_adapter_v1.safetensors',
    'zimage:turbo': 'zimage_turbo_training_adapter_v2.safetensors',
    minimax_h3: 'minimax_h3_training_adapter_v3.safetensors',
    minimax_h3_ref2va: 'minimax_h3_ref2va_training_adapter_v2.safetensors',
    p_h3_image: 'p_h3_image_training_adapter_v1.safetensors',
    minimax_h3_omni: 'fastvideo_fasth3_omni_training_adapter_v1.safetensors',
    minimax_h3_vsa: 'fastvideo_fasth3_8step_v2_training_adapter_v1.safetensors',
    'krea2:turbo': 'krea2_turbo_training_adapter_v1.safetensors',
    'krea2:o_edit_turbo': 'krea2_turbo_training_adapter_v1.safetensors',
  };
  for (const [arch, file] of Object.entries(adapters)) {
    assert.equal(diffusion.get(arch).get('config.process[0].model.assistant_lora_path'),
      `/poddata/ai-toolkit-models/loras/training_adapters/${file}`);
  }
  assert.equal(diffusion.get('ideogram4').get('config.process[0].model.unconditional_lora_path'),
    '/poddata/ai-toolkit-models/loras/ideogram_4_unconditional_lora_r16.safetensors');
  const text = source('extensions_built_in/diffusion_models/ui.tsx');
  assert.doesNotMatch(text, /["']ostris\/minimax_h3_training_adapter\//);
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
