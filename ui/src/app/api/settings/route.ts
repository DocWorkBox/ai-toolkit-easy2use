import { NextResponse } from 'next/server';
import prisma from '@/server/prisma';
import { normalizeTrainingSetting, normalizeDatasetsSetting, defaultModelsFolder } from '@/paths';
import { flushCache } from '@/server/settings';
import { settingsError } from '@/server/settingsError';

export async function GET() {
  try {
    const settings = await prisma.settings.findMany();
    const settingsObject = settings.reduce((acc: any, setting) => {
      acc[setting.key] = setting.value;
      return acc;
    }, {});
    settingsObject.TRAINING_FOLDER = normalizeTrainingSetting(settingsObject.TRAINING_FOLDER);
    settingsObject.DATASETS_FOLDER = normalizeDatasetsSetting(settingsObject.DATASETS_FOLDER);
    // MODELS_PATH from the env file always takes precedence over the setting
    if (process.env.MODELS_PATH && process.env.MODELS_PATH.trim() !== '') {
      settingsObject.MODELS_PATH = process.env.MODELS_PATH;
    } else if (!settingsObject.MODELS_PATH || settingsObject.MODELS_PATH === '') {
      // if MODELS_PATH is not set, use default
      settingsObject.MODELS_PATH = defaultModelsFolder;
    }
    return NextResponse.json(settingsObject);
  } catch (error) {
    const detail = settingsError(error);
    console.error('[settings:read]', detail);
    return NextResponse.json({ error: `读取设置失败：${detail}` }, { status: 500 });
  }
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const keys = ['HF_TOKEN', 'TRAINING_FOLDER', 'DATASETS_FOLDER', 'MODELS_PATH'];
    const invalid = keys.filter(key => typeof body?.[key] !== 'string');
    if (invalid.length) {
      return NextResponse.json(
        { error: `设置数据不完整：${invalid.join(', ')} 必须为字符串；请重新读取设置后再保存` },
        { status: 400 },
      );
    }
    const { HF_TOKEN, TRAINING_FOLDER, DATASETS_FOLDER, MODELS_PATH } = body;

    // A failed write must not leave a mixture of old and new settings.
    await prisma.$transaction([
      prisma.settings.upsert({
        where: { key: 'HF_TOKEN' },
        update: { value: HF_TOKEN },
        create: { key: 'HF_TOKEN', value: HF_TOKEN },
      }),
      prisma.settings.upsert({
        where: { key: 'TRAINING_FOLDER' },
        update: { value: normalizeTrainingSetting(TRAINING_FOLDER) },
        create: { key: 'TRAINING_FOLDER', value: normalizeTrainingSetting(TRAINING_FOLDER) },
      }),
      prisma.settings.upsert({
        where: { key: 'DATASETS_FOLDER' },
        update: { value: normalizeDatasetsSetting(DATASETS_FOLDER) },
        create: { key: 'DATASETS_FOLDER', value: normalizeDatasetsSetting(DATASETS_FOLDER) },
      }),
      prisma.settings.upsert({
        where: { key: 'MODELS_PATH' },
        update: { value: MODELS_PATH },
        create: { key: 'MODELS_PATH', value: MODELS_PATH },
      }),
    ]);

    flushCache();

    return NextResponse.json({ success: true });
  } catch (error) {
    if (error instanceof SyntaxError) {
      return NextResponse.json({ error: '设置请求不是有效的 JSON，请刷新设置页后重试' }, { status: 400 });
    }
    const detail = settingsError(error);
    console.error('[settings:write]', detail);
    return NextResponse.json({ error: `保存设置失败：${detail}` }, { status: 500 });
  }
}
