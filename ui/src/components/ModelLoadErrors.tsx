'use client';

import { RefreshCw } from 'lucide-react';

export default function ModelLoadErrors({
  errors,
  onRetry,
}: {
  errors: string[];
  onRetry: () => Promise<void> | void;
}) {
  if (!errors.length) return null;
  return (
    <div role="alert" className="relative z-30 mb-4 border-l-2 border-red-500 pl-3 text-sm text-red-400 min-w-0">
      <p className="font-medium">部分模型列表加载失败</p>
      <ul className="mt-2 space-y-1 whitespace-pre-wrap break-words [overflow-wrap:anywhere]">
        {errors.map((error, index) => (
          <li key={index}>{error}</li>
        ))}
      </ul>
      <button
        type="button"
        onClick={() => void onRetry()}
        className="mt-2 inline-flex items-center gap-2 text-gray-200 hover:text-white"
      >
        <RefreshCw className="h-4 w-4" />
        重新加载
      </button>
    </div>
  );
}
