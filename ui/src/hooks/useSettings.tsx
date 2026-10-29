'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { apiClient } from '@/utils/api';
import { getApiErrorMessage } from '@/utils/errors';

export interface Settings {
  HF_TOKEN: string;
  TRAINING_FOLDER: string;
  DATASETS_FOLDER: string;
  MODELS_PATH: string;
}

export default function useSettings() {
  const [settings, setSettings] = useState({
    HF_TOKEN: '',
    TRAINING_FOLDER: '',
    DATASETS_FOLDER: '',
    MODELS_PATH: '',
  });
  const [isSettingsLoaded, setIsLoaded] = useState(false);
  const [settingsError, setSettingsError] = useState<string | null>(null);
  const requestId = useRef(0);
  const reloadSettings = useCallback(async () => {
    const current = ++requestId.current;
    setIsLoaded(false);
    setSettingsError(null);
    try {
      const { data } = await apiClient.get('/api/settings');
      if (current !== requestId.current) return;
      setSettings({
        HF_TOKEN: data.HF_TOKEN || '',
        TRAINING_FOLDER: data.TRAINING_FOLDER || '',
        DATASETS_FOLDER: data.DATASETS_FOLDER || '',
        MODELS_PATH: data.MODELS_PATH || '',
      });
      setIsLoaded(true);
    } catch (error) {
      if (current !== requestId.current) return;
      setSettingsError(getApiErrorMessage(error));
    }
  }, []);
  useEffect(() => {
    void reloadSettings();
    return () => {
      requestId.current++;
    };
  }, [reloadSettings]);

  return { settings, setSettings, isSettingsLoaded, settingsError, reloadSettings };
}
