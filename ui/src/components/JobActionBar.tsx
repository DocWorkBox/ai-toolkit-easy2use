import Link from 'next/link';
import { Eye, Trash2, Pen, Play, Pause, Cog, X } from 'lucide-react';
import { Button } from '@headlessui/react';
import { openConfirm } from '@/components/ConfirmModal';
import { Job } from '@prisma/client';
import { startJob, stopJob, deleteJob, getAvaliableJobActions, markJobAsStopped } from '@/utils/jobs';
import { startQueue } from '@/utils/queue';
import { Menu, MenuButton, MenuItem, MenuItems } from '@headlessui/react';
import { redirect } from 'next/navigation';
import { openCaptionDatasetModal } from '@/components/CaptionDatasetModal';

interface JobActionBarProps {
  job: Job;
  onRefresh?: () => void;
  afterDelete?: () => void;
  hideView?: boolean;
  className?: string;
  autoStartQueue?: boolean;
}

export default function JobActionBar({
  job,
  onRefresh,
  afterDelete,
  className,
  hideView,
  autoStartQueue = false,
}: JobActionBarProps) {
  const { canStart, canStop, canDelete, canEdit, canRemoveFromQueue } = getAvaliableJobActions(job);

  if (!afterDelete) afterDelete = onRefresh;

  return (
    <div className={`${className}`}>
      {canStart && (
        <Button
          onClick={async () => {
            if (!canStart) return;
            await startJob(job.id);
            // start the queue as well
            if (autoStartQueue) {
              await startQueue(job.gpu_ids);
            }
            if (onRefresh) onRefresh();
          }}
          className={`ml-2 opacity-100`}
        >
          <Play />
        </Button>
      )}
      {canRemoveFromQueue && (
        <Button
          onClick={async () => {
            if (!canRemoveFromQueue) return;
            await markJobAsStopped(job.id);
            if (onRefresh) onRefresh();
          }}
          className={`ml-2 opacity-100`}
        >
          <X />
        </Button>
      )}
      {canStop && (
        <Button
          onClick={() => {
            if (!canStop) return;
            openConfirm({
              title: '停止任务',
              message: `确定要停止任务 "${job.name}" 吗？稍后可以恢复。`,
              type: 'info',
              confirmText: '停止',
              onConfirm: async () => {
                await stopJob(job.id);
                if (onRefresh) onRefresh();
              },
            });
          }}
          className={`ml-2 opacity-100`}
        >
          <Pause />
        </Button>
      )}
      {!hideView && (
        <Link href={`/jobs/${job.id}`} className="ml-2 text-gray-200 hover:text-gray-100 inline-block">
          <Eye />
        </Link>
      )}
      {job.job_type === 'caption' && canEdit && (
        <div
          className="ml-2 hover:text-gray-100 inline-block cursor-pointer"
          onClick={() =>
            openCaptionDatasetModal(
              job.job_ref || '',
              () => {
                if (onRefresh) onRefresh();
              },
              { jobId: job.id },
            )
          }
        >
          <Pen />
        </div>
      )}
      {job.job_type === 'train' && canEdit && (
        <Link href={`/jobs/new?id=${job.id}`} className="ml-2 hover:text-gray-100 inline-block">
          <Pen />
        </Link>
      )}
      <Button
        onClick={() => {
          let message = `确定要删除任务 "${job.name}" 吗？这将从磁盘永久删除它。`;
          if (job.status === 'running') {
            message += ' 警告：任务正在运行中。请尽可能先停止它。';
          }
          openConfirm({
            title: '删除任务',
            message: message,
            type: 'warning',
            confirmText: '删除',
            onConfirm: async () => {
              if (job.status === 'running') {
                try {
                  await stopJob(job.id);
                } catch (e) {
                  console.error('Error stopping job before deleting:', e);
                }
              }
              await deleteJob(job.id);
              if (afterDelete) afterDelete();
            },
          });
        }}
        className={`ml-2 opacity-100`}
      >
        <Trash2 />
      </Button>
      <div className="border-r border-1 border-gray-700 ml-2 inline"></div>
      <Menu>
        <MenuButton className={'ml-2'}>
          <Cog />
        </MenuButton>
        <MenuItems anchor="bottom" className="bg-gray-900 border border-gray-700 rounded shadow-lg w-48 px-2 py-2 mt-4">
          {job.job_type === 'train' && (
            <MenuItem>
              <Link
                href={`/jobs/new?cloneId=${job.id}`}
                className="cursor-pointer px-4 py-1 hover:bg-gray-800 rounded block text-gray-200"
              >
                克隆任务
              </Link>
            </MenuItem>
          )}
          <MenuItem>
            <div
              className="cursor-pointer px-4 py-1 hover:bg-gray-800 rounded text-gray-200"
              onClick={() => {
                let message = `确定要强制标记此任务为已停止吗？如果状态卡住，这将把状态设为'stopped'。请仅在确认任务确实已停止时执行此操作。这**不会**停止正在运行的任务。`;
                openConfirm({
                  title: '强制标记为已停止',
                  message: message,
                  type: 'warning',
                  confirmText: '标记为已停止',
                  onConfirm: async () => {
                    await markJobAsStopped(job.id);
                    onRefresh && onRefresh();
                  },
                });
              }}
            >
              强制标记为已停止
            </div>
          </MenuItem>
        </MenuItems>
      </Menu>
    </div>
  );
}
