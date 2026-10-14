import path from 'path';

export const DEFAULT_TRAINING_FOLDER = './output';

export function normalizeTrainingSetting(
  value: string | null | undefined,
  toolkitRoot: string,
  paths: typeof path = path,
): string {
  const configured = value?.trim() || DEFAULT_TRAINING_FOLDER;
  const resolved = paths.resolve(toolkitRoot, configured);
  const defaultRoot = paths.resolve(toolkitRoot, 'output');
  const isDefault = paths.sep === '\\'
    ? resolved.toLowerCase() === defaultRoot.toLowerCase()
    : resolved === defaultRoot;
  return isDefault ? DEFAULT_TRAINING_FOLDER : configured;
}

export function resolveTrainingRoot(
  value: string | null | undefined,
  toolkitRoot: string,
  paths: typeof path = path,
): string {
  return paths.resolve(toolkitRoot, normalizeTrainingSetting(value, toolkitRoot, paths));
}
