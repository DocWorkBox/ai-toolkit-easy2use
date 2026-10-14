import path from 'path';

export const DEFAULT_DATASETS_FOLDER = './datasets';

export function normalizeDatasetsSetting(
  value: string | null | undefined,
  toolkitRoot: string,
  paths: typeof path = path,
): string {
  const configured = value?.trim() || DEFAULT_DATASETS_FOLDER;
  const resolved = paths.resolve(toolkitRoot, configured);
  const defaultRoot = paths.resolve(toolkitRoot, 'datasets');
  const isDefault = paths.sep === '\\'
    ? resolved.toLowerCase() === defaultRoot.toLowerCase()
    : resolved === defaultRoot;
  return isDefault ? DEFAULT_DATASETS_FOLDER : configured;
}

export function resolveDatasetsRoot(
  value: string | null | undefined,
  toolkitRoot: string,
  paths: typeof path = path,
): string {
  return paths.resolve(toolkitRoot, normalizeDatasetsSetting(value, toolkitRoot, paths));
}
