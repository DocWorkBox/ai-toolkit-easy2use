import prisma from '../prisma';

import { Job, Queue } from '@prisma/client';
import startJob from './startJob';
import { isProcessAlive, finishJobProcess } from '../jobLifecycle';

export default async function processQueue() {
  const queues: Queue[] = await prisma.queue.findMany({
    orderBy: {
      id: 'asc',
    },
  });

  for (const queue of queues) {
    const processes = await prisma.job.findMany({ where: { gpu_ids: queue.gpu_ids, pid: { not: null } } });
    let hasLiveProcess = false;
    for (const job of processes) {
      if (isProcessAlive(job.pid!)) hasLiveProcess = true;
      else await finishJobProcess(prisma, job.id, job.pid!);
    }
    if (!queue.is_running) {
      // stop any running jobs first
      const runningJobs: Job[] = await prisma.job.findMany({
        where: {
          status: 'running',
          stop: false,
          gpu_ids: queue.gpu_ids,
        },
      });

      for (const job of runningJobs) {
        console.log(`Stopping job ${job.id} on GPU(s) ${job.gpu_ids}`);
        await prisma.job.update({
          where: { id: job.id },
          data: {
            return_to_queue: true,
            info: 'Stopping job...',
          },
        });
      }
    }
    if (queue.is_running) {
      if (hasLiveProcess) continue;
      // first see if one is already running, status of running or stopping
      const runningJob: Job | null = await prisma.job.findFirst({
        where: {
          status: { in: ['running', 'stopping'] },
          gpu_ids: queue.gpu_ids,
        },
      });

      if (runningJob) {
        // already running, nothing to do
        continue; // skip to next queue
      } else {
        // find the next job in the queue
        const nextJob: Job | null = await prisma.job.findFirst({
          where: {
            status: 'queued',
            gpu_ids: queue.gpu_ids,
          },
          orderBy: {
            queue_position: 'asc',
          },
        });
        if (nextJob) {
          console.log(`Starting job ${nextJob.id} on GPU(s) ${nextJob.gpu_ids}`);
          await startJob(nextJob.id);
        } else {
          // Recheck atomically: a resume may have arrived since findFirst.
          await prisma.$transaction(async tx => {
            const occupied = await tx.job.count({
              where: {
                gpu_ids: queue.gpu_ids,
                OR: [{ status: { in: ['queued', 'running', 'stopping'] } }, { pid: { not: null } }],
              },
            });
            if (occupied) return;
            await tx.queue.updateMany({ where: { id: queue.id, is_running: true }, data: { is_running: false } });
          });
        }
      }
    }
  }
}
