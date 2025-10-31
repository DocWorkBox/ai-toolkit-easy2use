"use client";
import { useEffect, useState } from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';

/**
 * 固定在页面左侧中间的侧边栏折叠按钮（手机和电脑端均显示）
 * - 通过全局事件与 Sidebar 通信：
 *   - 点击触发 `aitk:toggleSidebar`，让 Sidebar 切换折叠状态
 *   - 监听 `aitk:sidebarState`，同步当前折叠状态以切换箭头方向
 *   - 首次挂载时发送 `aitk:requestSidebarState` 请求，Sidebar 返回一次状态
 */
export default function SidebarToggleButton() {
  const [isCollapsed, setIsCollapsed] = useState(false);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    const onState = (e: Event) => {
      const custom = e as CustomEvent<{ collapsed: boolean }>;
      if (custom.detail && typeof custom.detail.collapsed === 'boolean') {
        setIsCollapsed(custom.detail.collapsed);
      }
    };
    window.addEventListener('aitk:sidebarState', onState as EventListener);
    // 请求一次当前状态
    window.dispatchEvent(new CustomEvent('aitk:requestSidebarState'));
    return () => {
      window.removeEventListener('aitk:sidebarState', onState as EventListener);
    };
  }, []);

  const toggle = () => {
    if (typeof window === 'undefined') return;
    window.dispatchEvent(new CustomEvent('aitk:toggleSidebar'));
  };

  return (
    <button
      suppressHydrationWarning
      // Localize accessible labels to English for consistency with the project
      aria-label={isCollapsed ? 'Expand Sidebar' : 'Collapse Sidebar'}
      title={isCollapsed ? 'Expand Sidebar' : 'Collapse Sidebar'}
      onClick={toggle}
      // 仅在移动端显示，桌面端隐藏
      className="fixed left-2 top-1/2 -translate-y-1/2 z-50 w-9 h-9 rounded-full bg-gray-800 border border-gray-700 shadow flex items-center justify-center text-gray-200 hover:bg-gray-700 transition-all duration-200 hover:scale-105 active:scale-95 md:hidden"
    >
      {isCollapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
    </button>
  );
}