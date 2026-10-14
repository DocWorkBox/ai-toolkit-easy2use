import path from 'path';
import prisma from './prisma';
import { DEFAULT_DATASETS_FOLDER, resolveDatasetsRoot as resolveDatasetPath } from './datasetPaths';
import { DEFAULT_TRAINING_FOLDER, resolveTrainingRoot as resolveTrainingPath } from './trainingPaths';

export const TOOLKIT_ROOT = path.resolve('@', '..', '..');
export const defaultTrainFolder = DEFAULT_TRAINING_FOLDER;
export const resolveTrainingRoot = (value?: string | null) => resolveTrainingPath(value, TOOLKIT_ROOT);
export const defaultDatasetsFolder = DEFAULT_DATASETS_FOLDER;
export const resolveDatasetsRoot = (value?: string | null) => resolveDatasetPath(value, TOOLKIT_ROOT);
export const defaultDataRoot = path.join(TOOLKIT_ROOT, 'data');
export const defaultModelsFolder = path.join(TOOLKIT_ROOT, 'models');

// Forked file-server workers set AI_TOOLKIT_QUIET_PATHS so this line prints
// once per launched process group, not once per worker.
if (!process.env.AI_TOOLKIT_QUIET_PATHS) {
  console.log('TOOLKIT_ROOT:', TOOLKIT_ROOT);
}

export const getTrainingFolder = async () => {
  const key = 'TRAINING_FOLDER';
  let row = await prisma.settings.findFirst({
    where: {
      key: key,
    },
  });
  let trainingRoot = defaultTrainFolder;
  if (row?.value && row.value !== '') {
    trainingRoot = row.value;
  }
  return resolveTrainingRoot(trainingRoot);
};

export const getHFToken = async () => {
  const key = 'HF_TOKEN';
  let row = await prisma.settings.findFirst({
    where: {
      key: key,
    },
  });
  let token = '';
  if (row?.value && row.value !== '') {
    token = row.value;
  }
  return token;
};

export const getModelsPath = async () => {
  const key = 'MODELS_PATH';
  let row = await prisma.settings.findFirst({
    where: {
      key: key,
    },
  });
  let modelsPath = '';
  if (row?.value && row.value !== '' && row.value !== defaultModelsFolder) {
    modelsPath = row.value;
  }
  return modelsPath;
};
