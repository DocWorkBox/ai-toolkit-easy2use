import type { PrismaClient } from '@prisma/client';

export function isProcessAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch (error: any) {
    return error?.code !== 'ESRCH';
  }
}

// Python can write its final status before the OS process finishes exiting.
// Retain the PID until then, and never let an old observer update a new run.
export async function finishJobProcess(
  prisma: PrismaClient,
  jobID: string,
  pid: number,
  message = 'Job process exited unexpectedly.',
) {
  if (isProcessAlive(pid)) return { count: 0, unexpected: false };
  const job = await prisma.job.findUnique({ where: { id: jobID } });
  if (!job || job.pid !== pid) return { count: 0, unexpected: false };
  const active = ['running', 'stopping'].includes(job.status);
  const unexpected = active && !job.stop && !job.return_to_queue;
  const data: { pid: null; status?: string; info?: string } = { pid: null };
  if (active) {
    data.status = job.stop ? 'stopped' : job.return_to_queue ? 'queued' : 'error';
    data.info = job.stop ? 'Job stopped' : job.return_to_queue ? 'Job queued' : message;
  }
  const result = await prisma.job.updateMany({
    where: { id: jobID, pid, status: job.status, stop: job.stop, return_to_queue: job.return_to_queue },
    data,
  });
  return { count: result.count, unexpected: unexpected && result.count > 0 };
}
