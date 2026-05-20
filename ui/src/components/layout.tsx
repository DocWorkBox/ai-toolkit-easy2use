'use client';
import classNames from 'classnames';
import ThemeLogo from './ThemeLogo';
import { mobileSidebarState } from './Sidebar';

interface Props {
  className?: string;
  children?: React.ReactNode;
}

const MobileMenuButton: React.FC = () => {
  const [, setIsMobileOpen] = mobileSidebarState.use();
  return (
    <button
      onClick={() => setIsMobileOpen(true)}
      className="md:hidden flex items-center ml-2 mr-1 px-1 py-1 rounded-md hover:bg-gray-800"
      aria-label="Open menu"
    >
      <ThemeLogo />
    </button>
  );
};

export const TopBar: React.FC<Props> = ({ children, className }) => {
  return (
    <div
      // 抑制首帧 SSR/CSR 的极小结构差异引发的水合警告
      suppressHydrationWarning
      className={classNames(
        // 顶部栏：禁止横向滚动，保持不换行，靠限制元素宽度避免溢出
        'absolute top-0 left-0 w-full h-12 bg-gray-900 shadow-sm z-10 flex items-center px-2 gap-2 overflow-hidden whitespace-nowrap',
        className,
      )}
    >
      <MobileMenuButton />
      {children ? children : null}
    </div>
  );
};

export const MainContent: React.FC<Props> = ({ children, className }) => {
  return (
    <div className={classNames('pt-14 px-2 sm:px-4 absolute top-0 left-0 w-full h-full overflow-auto', className)}>
      {children ? children : null}
    </div>
  );
};
