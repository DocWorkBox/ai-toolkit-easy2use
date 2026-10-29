import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/server/prisma';
import { isProcessAlive, finishJobProcess } from '../../../../../../cron/jobLifecycle';

export async function GET(request: NextRequest, { params }: { params: Promise<{ jobID: string }> }) {
  const { jobID } = await params;

  let job = await prisma.job.findUnique({
    where: { id: jobID },
  });

  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }
  if (job.pid != null) {
    if (isProcessAlive(job.pid)) {
      return NextResponse.json({ error: '上一次任务进程仍在运行或退出中，请等待停止完成后再启动' }, { status: 409 });
    }
    await finishJobProcess(prisma, jobID, job.pid);
    job = await prisma.job.findUnique({ where: { id: jobID } });
  }
  if (!job) return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  if (job.status === 'queued') return NextResponse.json(job);
  if (!['stopped', 'error', 'completed'].includes(job.status)) {
    return NextResponse.json({ error: '任务正在启动或停止中，请等待状态更新' }, { status: 409 });
  }
  const resumeStoppedJob = job.stop;

  // get highest queue position
  const highestQueuePosition = await prisma.job.aggregate({
    _max: {
      queue_position: true,
    },
  });
  const newQueuePosition = (highestQueuePosition._max.queue_position || 0) + 1000;

  const snapshot = job;
  const queued = await prisma.$transaction(async tx => {
    const result = await tx.job.updateMany({
      where: { id: jobID, status: snapshot.status, pid: null, stop: snapshot.stop },
      data: {
        queue_position: newQueuePosition,
        status: 'queued',
        stop: false,
        return_to_queue: false,
        info: 'Job queued',
      },
    });
    if (result.count === 0) return result;
    // Resuming an explicitly stopped job must also wake its idle queue.
    // Newly queued jobs keep the existing manual queue-start behavior.
    await tx.queue.upsert({
      where: { gpu_ids: snapshot.gpu_ids },
      create: { gpu_ids: snapshot.gpu_ids, is_running: resumeStoppedJob },
      update: resumeStoppedJob ? { is_running: true } : {},
    });
    return result;
  });
  if (queued.count === 0) {
    return NextResponse.json({ error: '任务状态已变化，请刷新后重试' }, { status: 409 });
  }

  // Return the response immediately
  return NextResponse.json(await prisma.job.findUnique({ where: { id: jobID } }));
}
