import classNames from 'classnames';

interface Props {
  className?: string;
  children?: React.ReactNode;
}

export const TopBar: React.FC<Props> = ({ children, className }) => {
  return (
    <div
      // 抑制首帧 SSR/CSR 的极小结构差异引发的水合警告
      suppressHydrationWarning
      className={classNames(
        // 顶部栏：禁止横向滚动，保持不换行，靠限制元素宽度避免溢出
        'absolute top-0 left-0 w-full h-12 dark:bg-gray-900 shadow-sm z-10 flex items-center px-2 gap-2 overflow-hidden whitespace-nowrap',
        className,
      )}
    >
      {children ? children : null}
    </div>
  );
};

export const MainContent: React.FC<Props> = ({ children, className }) => {
  return (
    <div className={classNames('pt-14 px-4 absolute top-0 left-0 w-full h-full overflow-auto', className)}>
      {children ? children : null}
    </div>
  );
};
