'use client';

import { useState } from 'react';
import useSettings from '@/hooks/useSettings';
import { TopBar, MainContent } from '@/components/layout';
import { apiClient } from '@/utils/api';
import { getApiErrorMessage } from '@/utils/errors';
import { RefreshCw } from 'lucide-react';

export default function Settings() {
  const { settings, setSettings, isSettingsLoaded, settingsError, reloadSettings } = useSettings();
  const [status, setStatus] = useState<'idle' | 'saving' | 'success' | 'error'>('idle');
  const [saveError, setSaveError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!isSettingsLoaded || status === 'saving') return;
    setStatus('saving');
    setSaveError(null);
    try {
      await apiClient.post('/api/settings', settings);
      setStatus('success');
    } catch (error) {
      setSaveError(getApiErrorMessage(error));
      setStatus('error');
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setSettings(prev => ({ ...prev, [name]: value }));
    setStatus('idle');
    setSaveError(null);
  };

  return (
    <>
      <TopBar>
        <div>
          <h1 className="text-base sm:text-lg">设置</h1>
        </div>
        <div className="flex-1"></div>
      </TopBar>
      <MainContent>
        {settingsError && (
          <div role="alert" className="mb-4 text-red-400 break-words">
            <p>读取设置失败：{settingsError}</p>
            <button type="button" onClick={() => void reloadSettings()} className="mt-2 flex items-center gap-2 text-gray-200">
              <RefreshCw className="w-4 h-4" />重试
            </button>
          </div>
        )}
        {!isSettingsLoaded && !settingsError && <p role="status" className="mb-4 text-gray-400">正在读取设置...</p>}
        <form onSubmit={handleSubmit} className="space-y-6">
          <fieldset disabled={!isSettingsLoaded || status === 'saving'} className="grid grid-cols-1 gap-6 sm:grid-cols-2">
            <div>
              <div className="space-y-4">
                <div>
                  <label htmlFor="HF_TOKEN" className="block text-sm font-medium mb-2">
                    Hugging Face 令牌
                    <div className="text-gray-500 text-sm ml-1">
                      如果需要访问受限/私有模型，请在{' '}
                      <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noreferrer">
                        Huggingface
                      </a>{' '}
                      创建一个 Read Token（读取令牌）。
                    </div>
                  </label>
                  <input
                    type="password"
                    id="HF_TOKEN"
                    name="HF_TOKEN"
                    value={settings.HF_TOKEN}
                    onChange={handleChange}
                    className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-gray-600 focus:border-transparent"
                    placeholder="输入 Hugging Face 令牌"
                  />
                </div>

                <div>
                  <label htmlFor="TRAINING_FOLDER" className="block text-sm font-medium mb-2">
                    训练输出路径
                    <div className="text-gray-500 text-sm ml-1">
                      默认使用 ./output，相对于训练器根目录；也支持自定义绝对路径。留空时使用默认目录。
                    </div>
                  </label>
                  <input
                    type="text"
                    id="TRAINING_FOLDER"
                    name="TRAINING_FOLDER"
                    value={settings.TRAINING_FOLDER}
                    onChange={handleChange}
                    className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-gray-600 focus:border-transparent"
                    placeholder="输入训练输出路径"
                  />
                </div>

                <div>
                  <label htmlFor="DATASETS_FOLDER" className="block text-sm font-medium mb-2">
                    数据集目录路径
                    <div className="text-gray-500 text-sm ml-1">
                      存放与读取数据集的目录。
                      <span className="text-orange-800">
                        警告：本软件可能会修改数据集，建议在其他位置保留备份，或为本软件单独准备一个专用目录。
                      </span>
                    </div>
                  </label>
                  <input
                    type="text"
                    id="DATASETS_FOLDER"
                    name="DATASETS_FOLDER"
                    value={settings.DATASETS_FOLDER}
                    onChange={handleChange}
                    className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-gray-600 focus:border-transparent"
                    placeholder="输入数据集目录路径"
                  />
                </div>

                <div>
                  <label htmlFor="MODELS_PATH" className="block text-sm font-medium mb-2">
                    模型目录路径
                    <div className="text-gray-500 text-sm ml-1">
                      部分模型支持直接加载 ComfyUI 格式权重，这些权重会从该目录加载或下载到该目录。
                      必须填写绝对路径；留空时默认使用项目根目录下的 models 文件夹。
                    </div>
                  </label>
                  <input
                    type="text"
                    id="MODELS_PATH"
                    name="MODELS_PATH"
                    value={settings.MODELS_PATH}
                    onChange={handleChange}
                    className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-gray-600 focus:border-transparent"
                    placeholder="输入模型目录路径"
                  />
                </div>
              </div>
            </div>
          </fieldset>

          <button
            type="submit"
            disabled={!isSettingsLoaded || status === 'saving'}
            className="w-full px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {status === 'saving' ? '正在保存...' : '保存设置'}
          </button>

          {status === 'success' && <p className="text-green-500 text-center">设置保存成功！</p>}
          {status === 'error' && <p role="alert" className="text-red-400 break-words">{saveError}</p>}
        </form>
      </MainContent>
    </>
  );
}
