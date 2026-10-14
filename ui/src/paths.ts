import path from 'path';
import {
  DEFAULT_DATASETS_FOLDER,
  normalizeDatasetsSetting as normalizeDatasetPath,
  resolveDatasetsRoot as resolveDatasetPath,
} from '../cron/datasetPaths';
export const TOOLKIT_ROOT = path.resolve('@', '..', '..');
export const defaultTrainFolder = path.join(TOOLKIT_ROOT, 'output');
export const defaultDatasetsFolder = DEFAULT_DATASETS_FOLDER;
export const normalizeDatasetsSetting = (value?: string | null) => normalizeDatasetPath(value, TOOLKIT_ROOT);
export const resolveDatasetsRoot = (value?: string | null) => resolveDatasetPath(value, TOOLKIT_ROOT);
export const defaultDataRoot = path.join(TOOLKIT_ROOT, 'data');
export const defaultModelsFolder = path.join(TOOLKIT_ROOT, 'models');
