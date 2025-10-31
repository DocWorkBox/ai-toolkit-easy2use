'use client';

import JobsTable from '@/components/JobsTable';
import { TopBar, MainContent } from '@/components/layout';
import Link from 'next/link';

export default function Dashboard() {
  return (
    <>
      <TopBar>
        <div>
          {/* 标题在移动端缩小并截断 */}
          <h1 className="text-base sm:text-lg truncate max-w-[50vw] sm:max-w-none">Training Queue</h1>
        </div>
        <div className="flex-1"></div>
        <div>
          {/* 右侧按钮在移动端缩小尺寸并保持不换行 */}
          <Link href="/jobs/new" className="text-gray-200 bg-slate-600 text-xs sm:text-sm px-2 sm:px-3 py-1 rounded-md whitespace-nowrap">
            New Training Job
          </Link>
        </div>
      </TopBar>
      <MainContent>
        <JobsTable />
      </MainContent>
    </>
  );
}
