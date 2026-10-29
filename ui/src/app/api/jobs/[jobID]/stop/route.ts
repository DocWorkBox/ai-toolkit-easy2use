import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/server/prisma';
import { exec } from 'child_process';
import { promisify } from 'util';
import { isProcessAlive, finishJobProcess } from '../../../../../../cron/jobLifecycle';

const execAsync = promisify(exec);
const isWindows = process.platform === 'win32';

// How long a stopping job gets to run its graceful shutdown (KeyboardInterrupt
// -> on_error -> final DB write) before we assume it is hung and force-kill it.
const GRACEFUL_STOP_TIMEOUT_MS = 60_000;
const BACKSTOP_POLL_MS = 2_000;

// Windows: the trainer's stop watcher sees the `stop` flag and raises SIGINT
// inside the process, so the graceful path runs without us sending anything.
// This backstop only exists for a job that is too hung to notice the flag.
// Poll (rather than one long timer) so we stop watching the moment the pid
// dies and never touch a recycled pid.
const scheduleForceKillBackstop = (pid: number, jobID: string) => {
  const startedAt = Date.now();
  let checking = false;
  const timer = setInterval(async () => {
    if (checking) return;
    checking = true;
    try {
      if (!isProcessAlive(pid)) {
        clearInterval(timer);
        await finishJobProcess(prisma, jobID, pid);
        return;
      }
      const job = await prisma.job.findUnique({ where: { id: jobID } });
      if (!job || job.pid !== pid || !job.stop) {
        clearInterval(timer);
        return;
      }
      if (Date.now() - startedAt < GRACEFUL_STOP_TIMEOUT_MS) return;
      clearInterval(timer);
      console.warn(
        `Job ${jobID} (pid ${pid}) still alive ${GRACEFUL_STOP_TIMEOUT_MS / 1000}s after stop request, force killing`,
      );
      execAsync(`taskkill /PID ${pid} /T /F`, { windowsHide: true }).catch(() => {
        // already gone
      });
    } catch (error) {
      console.error('Error checking stopping job:', error);
    } finally {
      checking = false;
    }
  }, BACKSTOP_POLL_MS);
  timer.unref?.();
};

export async function GET(request: NextRequest, { params }: { params: Promise<{ jobID: string }> }) {
  const { jobID } = await params;

  const job = await prisma.job.findUnique({
    where: { id: jobID },
  });

  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }

  const stillExiting = job.pid != null ? isProcessAlive(job.pid) : ['running', 'stopping'].includes(job.status);
  const result = await prisma.job.updateMany({
    where: { id: jobID, pid: job.pid, status: job.status },
    data: {
      stop: true,
      return_to_queue: false,
      status: stillExiting ? 'stopping' : 'stopped',
      info: stillExiting ? 'Stopping job...' : 'Job stopped',
      ...(!stillExiting ? { pid: null } : {}),
    },
  });
  if (result.count === 0) {
    return NextResponse.json({ error: '任务状态已变化，请刷新后重试' }, { status: 409 });
  }

  // Send SIGINT to the process if we have a PID
  if (job.pid != null && stillExiting) {
    console.log(`Attempting to stop job ${jobID} with PID ${job.pid}`);
    try {
      if (isWindows) {
        // No external SIGINT possible on Windows (the job runs under pythonw
        // with no console), and none is needed: the `stop` flag written above
        // is the signal. The trainer's stop watcher polls it and raises
        // SIGINT in-process, which runs the same KeyboardInterrupt shutdown
        // as Linux -- final DB write and the closing log lines.
        scheduleForceKillBackstop(job.pid, jobID);
      } else {
        process.kill(job.pid, 'SIGINT');
      }
    } catch (e) {
      // Process may have already exited — that's fine
      console.error('Error sending signal to process:', e);
    }
  } else {
    console.warn(`No PID found for job ${jobID}, cannot send stop signal`);
  }

  return NextResponse.json(await prisma.job.findUnique({ where: { id: jobID } }));
}
