import { useMemo, useState, useRef, useEffect } from 'react';
import useSampleImages from '@/hooks/useSampleImages';
import SampleImageCard from './SampleImageCard';
import { Job } from '@prisma/client';
import { JobConfig } from '@/types';
import { LuImageOff, LuLoader, LuBan } from 'react-icons/lu';
import { Button } from '@headlessui/react';
import { FaDownload } from 'react-icons/fa';
import { apiClient } from '@/utils/api';
import classNames from 'classnames';
import { FaCaretDown, FaCaretUp } from 'react-icons/fa';
import SampleImageViewer from './SampleImageViewer';

interface SampleImagesMenuProps {
  job?: Job | null;
}

export const SampleImagesMenu = ({ job }: SampleImagesMenuProps) => {
  const [isZipping, setIsZipping] = useState(false);

  const downloadZip = async () => {
    if (isZipping) return;
    setIsZipping(true);

    try {
      const res = await apiClient.post('/api/zip', {
        zipTarget: 'samples',
        jobName: job?.name,
      });

      const zipPath = res.data.zipPath; // e.g. /mnt/Train2/out/ui/.../samples.zip
      if (!zipPath) throw new Error('No zipPath in response');

      const downloadPath = `/api/files/${encodeURIComponent(zipPath)}`;
      const a = document.createElement('a');
      a.href = downloadPath;
      // optional: suggest filename (browser may ignore if server sets Content-Disposition)
      a.download = 'samples.zip';
      document.body.appendChild(a);
      a.click();
      a.remove();
    } catch (err) {
      console.error('Error downloading zip:', err);
    } finally {
      setIsZipping(false);
    }
  };
  return (
    <Button
      onClick={downloadZip}
      className={classNames(`px-4 py-1 h-8 hover:bg-gray-200 dark:hover:bg-gray-700`, {
        'opacity-50 cursor-not-allowed': isZipping,
      })}
    >
      {isZipping ? (
        <LuLoader className="animate-spin inline-block mr-2" />
      ) : (
        <FaDownload className="inline-block mr-2" />
      )}
      {isZipping ? 'Preparing' : 'Download'}
    </Button>
  );
};

interface SampleImagesProps {
  job: Job;
}

export default function SampleImages({ job }: SampleImagesProps) {
  const { sampleImages, status, refreshSampleImages } = useSampleImages(job.id, 5000);
  const [selectedSamplePath, setSelectedSamplePath] = useState<string | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const didFirstScroll = useRef(false);
  const numSamples = useMemo(() => {
    if (job?.job_config) {
      const jobConfig = JSON.parse(job.job_config) as JobConfig;
      const sampleConfig = jobConfig.config.process[0].sample;
      const numPrompts = sampleConfig.prompts ? sampleConfig.prompts.length : 0;
      const numSamples = sampleConfig.samples.length;
      return Math.max(numPrompts, numSamples, 1);
    }
    return 10;
  }, [job]);

  const scrollToBottom = () => {
    if (containerRef.current) {
      containerRef.current.scrollTo({ top: containerRef.current.scrollHeight, behavior: 'instant' });
    }
  };

  const scrollToTop = () => {
    if (containerRef.current) {
      containerRef.current.scrollTo({ top: 0, behavior: 'instant' });
    }
  };

  const PageInfoContent = useMemo(() => {
    let icon = null;
    let text = '';
    let subtitle = '';
    let showIt = false;
    let bgColor = '';
    let textColor = '';
    let iconColor = '';

    if (sampleImages.length > 0) return null;

    if (status == 'loading') {
      icon = <LuLoader className="animate-spin w-8 h-8" />;
      text = 'Loading Samples';
      subtitle = 'Please wait while we fetch your samples...';
      showIt = true;
      bgColor = 'bg-gray-50 dark:bg-gray-800/50';
      textColor = 'text-gray-900 dark:text-gray-100';
      iconColor = 'text-gray-500 dark:text-gray-400';
    }
    if (status == 'error') {
      icon = <LuBan className="w-8 h-8" />;
      text = 'Error Loading Samples';
      subtitle = 'There was a problem fetching the samples.';
      showIt = true;
      bgColor = 'bg-red-50 dark:bg-red-950/20';
      textColor = 'text-red-900 dark:text-red-100';
      iconColor = 'text-red-600 dark:text-red-400';
    }
    if (status == 'success' && sampleImages.length === 0) {
      icon = <LuImageOff className="w-8 h-8" />;
      text = 'No Samples Found';
      subtitle = 'No samples have been generated yet';
      showIt = true;
      bgColor = 'bg-gray-50 dark:bg-gray-800/50';
      textColor = 'text-gray-900 dark:text-gray-100';
      iconColor = 'text-gray-500 dark:text-gray-400';
    }

    if (!showIt) return null;

    return (
      <div
        className={`mt-10 flex flex-col items-center justify-center py-16 px-8 rounded-xl border-2 border-gray-700 border-dashed ${bgColor} ${textColor} mx-auto max-w-md text-center`}
      >
        <div className={`${iconColor} mb-4`}>{icon}</div>
        <h3 className="text-lg font-semibold mb-2">{text}</h3>
        <p className="text-sm opacity-75 leading-relaxed">{subtitle}</p>
      </div>
    );
  }, [status, sampleImages.length]);

  // 将样例图栅格改为响应式：
  // - 手机端两列（更易于浏览）
  // - 随屏幕大小逐步增加列数，但限制为最多 8 列
  // 直接返回明确的 Tailwind 类，避免被构建时清理
  const gridColsClass = useMemo(() => {
    // 当每次采样只生成1张图片时，默认从单列开始，避免出现大面积空白
    const cols = Math.min(numSamples, 8);
    const map: { [key: number]: string } = {
      1: 'grid-cols-1 sm:grid-cols-2 md:grid-cols-3',
      2: 'grid-cols-2 sm:grid-cols-3 md:grid-cols-4',
      3: 'grid-cols-2 sm:grid-cols-3 md:grid-cols-4',
      4: 'grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5',
      5: 'grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6',
      6: 'grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 2xl:grid-cols-7',
      7: 'grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 xl:grid-cols-7',
      8: 'grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 xl:grid-cols-8',
    };
    return map[cols] || 'grid-cols-2 sm:grid-cols-3 md:grid-cols-4';
  }, [numSamples]);

  const sampleConfig = useMemo(() => {
    if (job?.job_config) {
      const jobConfig = JSON.parse(job.job_config) as JobConfig;
      return jobConfig.config.process[0].sample;
    }
    return null;
  }, [job]);

  // scroll to bottom on first load of samples
  useEffect(() => {
    if (status === 'success' && sampleImages.length > 0 && !didFirstScroll.current) {
      didFirstScroll.current = true;
      setTimeout(() => {
        scrollToBottom();
      }, 100);
    }
  }, [status, sampleImages.length]);

  return (
    <div ref={containerRef} className="absolute top-[80px] left-0 right-0 bottom-0 overflow-y-auto">
      <div className="pb-4">
        {PageInfoContent}
        {sampleImages && (
          <div className={`grid ${gridColsClass} gap-1`}>
            {sampleImages.map((sample: string, idx: number) => {
              // Compute current group (groups are size = numSamples)
              const groupIndex = Math.floor(idx / numSamples);
              const groupStart = groupIndex * numSamples;
              const groupEnd = Math.min(groupStart + numSamples, sampleImages.length);
              const groupSize = groupEnd - groupStart;
              const isEndOfGroup = idx === groupEnd - 1;

              // Only enforce a MIN of 3 when the group's planned width is < 3
              // 当单张采样时不进行占位填充，避免“每次采样之间有两格空白”的视觉问题
              const MIN_COLS = numSamples <= 1 ? 1 : 3;
              const shouldPad = numSamples < MIN_COLS && groupSize < MIN_COLS;
              const padsNeeded = shouldPad ? MIN_COLS - groupSize : 0;

              return (
                <div key={sample} className="contents">
                  <SampleImageCard
                    imageUrl={sample}
                    numSamples={numSamples}
                    sampleImages={sampleImages}
                    alt="Sample Image"
                    onClick={() => setSelectedSamplePath(sample)}
                    observerRoot={containerRef.current}
                  />

                  {isEndOfGroup &&
                    padsNeeded > 0 &&
                    Array.from({ length: padsNeeded }).map((_, i) => (
                      <div key={`pad-${groupIndex}-${i}`} className="invisible" />
                    ))}
                </div>
              );
            })}
          </div>
        )}
      </div>
      <SampleImageViewer
        imgPath={selectedSamplePath}
        numSamples={numSamples}
        sampleImages={sampleImages}
        onChange={setPath => setSelectedSamplePath(setPath)}
        sampleConfig={sampleConfig}
        refreshSampleImages={refreshSampleImages}
      />
      <div
        className="fixed top-20 mt-4 right-6 w-10 h-10 rounded-full bg-gray-900 shadow-lg flex items-center justify-center text-white opacity-80 hover:opacity-100 cursor-pointer"
        onClick={scrollToTop}
        title="Scroll to Top"
      >
        <FaCaretUp className="text-gray-500 dark:text-gray-400" />
      </div>
      <div
        className="fixed bottom-5 right-6 w-10 h-10 rounded-full bg-gray-900 shadow-lg flex items-center justify-center text-white opacity-80 hover:opacity-100 cursor-pointer"
        onClick={scrollToBottom}
        title="Scroll to Bottom"
      >
        <FaCaretDown className="text-gray-500 dark:text-gray-400" />
      </div>
    </div>
  );
}
