import React, { use, useEffect, useRef } from 'react';
import { Button } from '@headlessui/react';
import { CaptionDatasetModal, openCaptionDatasetModal } from '@/components/CaptionDatasetModal';
import useJobByRef from '@/hooks/useJobByRef';
import Link from 'next/link';
import { Loader2 } from 'lucide-react';

type AutoCaptionButtonProps = {
  datasetPath: string;
  setIsAutoCaptioning?: (isAutoCaptioning: boolean) => void;
  onCaptioningFinished?: () => void;
  captionExt?: string;
};

export default function AutoCaptionButton({
  datasetPath,
  setIsAutoCaptioning,
  onCaptioningFinished,
  captionExt,
}: AutoCaptionButtonProps) {
  const { job, status, refreshJob } = useJobByRef(datasetPath, 5000);
  const isActive = !!(job && (job.status === 'running' || job.status === 'queued'));
  const lastFinishedJobIdRef = useRef<string | null>(null);

  useEffect(() => {
    if (setIsAutoCaptioning) {
      setIsAutoCaptioning(isActive);
    }
  }, [isActive, setIsAutoCaptioning]);

  useEffect(() => {
    if (!job) return;
    if (isActive) {
      if (lastFinishedJobIdRef.current === job.id) {
        lastFinishedJobIdRef.current = null;
      }
      return;
    }
    if (
      ['completed', 'stopped', 'error'].includes(job.status) &&
      lastFinishedJobIdRef.current !== job.id
    ) {
      lastFinishedJobIdRef.current = job.id;
      onCaptioningFinished?.();
    }
  }, [job?.id, job?.status, isActive, onCaptioningFinished]);

  if (isActive && job) {
    return (
      <Link
        href={`/jobs/${job.id}`}
        className="text-white bg-gray-400 px-2 sm:px-3 py-1 rounded-md mr-1 sm:mr-2 inline-flex items-center gap-1 sm:gap-1.5 text-sm sm:text-base whitespace-nowrap"
      >
        <Loader2 className="w-4 h-4 animate-spin" />
        <span className="hidden sm:inline">正在自动打标...</span>
        <span className="sm:hidden">打标中</span>
      </Link>
    );
  }
  return (
    <Button
      className="text-white bg-blue-600 px-2 sm:px-3 py-1 rounded-md mr-1 sm:mr-2 text-sm sm:text-base whitespace-nowrap"
      onClick={() =>
        openCaptionDatasetModal(
          datasetPath,
          () => {
            refreshJob();
          },
          {
            defaultCaptionExt: captionExt,
            onJobStarted: () => setIsAutoCaptioning?.(true),
          },
        )
      }
    >
      <span className="hidden sm:inline">自动打标</span>
      <span className="sm:hidden">打标</span>
    </Button>
  );
}
