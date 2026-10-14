import path from 'path';
import {
  DEFAULT_DATASETS_FOLDER,
  normalizeDatasetsSetting as normalizeDatasetPath,
  resolveDatasetsRoot as resolveDatasetPath,
} from '../cron/datasetPaths';
import {
  DEFAULT_TRAINING_FOLDER,
  normalizeTrainingSetting as normalizeTrainingPath,
  resolveTrainingRoot as resolveTrainingPath,
} from '../cron/trainingPaths';
export const TOOLKIT_ROOT = path.resolve('@', '..', '..');
export const defaultTrainFolder = DEFAULT_TRAINING_FOLDER;
export const normalizeTrainingSetting = (value?: string | null) => normalizeTrainingPath(value, TOOLKIT_ROOT);
export const resolveTrainingRoot = (value?: string | null) => resolveTrainingPath(value, TOOLKIT_ROOT);
export const defaultDatasetsFolder = DEFAULT_DATASETS_FOLDER;
export const normalizeDatasetsSetting = (value?: string | null) => normalizeDatasetPath(value, TOOLKIT_ROOT);
export const resolveDatasetsRoot = (value?: string | null) => resolveDatasetPath(value, TOOLKIT_ROOT);
export const defaultDataRoot = path.join(TOOLKIT_ROOT, 'data');
export const defaultModelsFolder = path.join(TOOLKIT_ROOT, 'models');
