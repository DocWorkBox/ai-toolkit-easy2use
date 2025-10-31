"use client";
import Link from 'next/link';
import { Home, Settings, BrainCircuit, Images, Plus, Heart } from 'lucide-react';
import { FaYoutube } from 'react-icons/fa6';
import { SiBilibili } from 'react-icons/si';
import { useEffect, useState } from 'react';
import classNames from 'classnames';

const Sidebar = () => {
  const navigation = [
    { name: '仪表盘', href: '/dashboard', icon: Home },
    { name: '新建任务', href: '/jobs/new', icon: Plus },
    { name: '训练队列', href: '/jobs', icon: BrainCircuit },
    { name: '数据集', href: '/datasets', icon: Images },
    { name: '设置', href: '/settings', icon: Settings },
  ];

  const socialsBoxClass =
    'flex flex-col items-center justify-center p-1 hover:bg-gray-800 rounded-lg transition-colors';
  const socialIconClass = 'w-5 h-5 text-gray-400 hover:text-white';

  // 折叠状态（仅影响移动端）
  // 默认在小屏折叠，桌面端由响应式类控制始终展开。
  // 为避免 SSR 与客户端首屏不一致导致的 hydration 警告，这里不在首次渲染时依赖 window。
  const [isOpenOnMobile, setIsOpenOnMobile] = useState(false);

  // 切换折叠状态
  const toggleCollapse = () => setIsOpenOnMobile(c => !c);

  // 监听来自 TopBar 的全局事件，以便在顶部栏中控制侧边栏折叠
  useEffect(() => {
    if (typeof window === 'undefined') return;
    const handler = () => toggleCollapse();
    window.addEventListener('aitk:toggleSidebar', handler as EventListener);
    return () => window.removeEventListener('aitk:toggleSidebar', handler as EventListener);
  }, []);

  // 对外广播当前折叠状态（首次与每次变更），并响应状态请求
  useEffect(() => {
    if (typeof window === 'undefined') return;
    const collapsed = window.innerWidth < 768 ? !isOpenOnMobile : false;
    window.dispatchEvent(new CustomEvent('aitk:sidebarState', { detail: { collapsed } }));
  }, [isOpenOnMobile]);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    const onRequest = () => {
      const collapsed = window.innerWidth < 768 ? !isOpenOnMobile : false;
      window.dispatchEvent(new CustomEvent('aitk:sidebarState', { detail: { collapsed } }));
    };
    window.addEventListener('aitk:requestSidebarState', onRequest as EventListener);
    return () => window.removeEventListener('aitk:requestSidebarState', onRequest as EventListener);
  }, [isOpenOnMobile]);

  const AvatarOrHeart = () => {
    const [useHeartIcon, setUseHeartIcon] = useState(false);
    const [srcIndex, setSrcIndex] = useState(0);
    const candidates = [
      // 先尝试 .jpg，避免默认 .png 产生 404
      '/doc_workbox_avatar.jpg',
      '/doc_workbox_avatar.png',
      '/doc_workbox_avatar.jpeg',
      '/doc_workbox_avatar.webp',
      '/doc_workbox_avatar.svg',
      '/doc_workbox_avatar.avif'
    ];

    if (useHeartIcon) {
      return <Heart className="w-6 h-6 text-pink-400" aria-label="Doc_workBox 爱心" />;
    }

    return (
      <img
        src={candidates[srcIndex]}
        alt="Doc_workBox 头像"
        className="w-6 h-6 rounded object-cover"
        onError={() => {
          const next = srcIndex + 1;
          if (next < candidates.length) {
            setSrcIndex(next);
          } else {
            setUseHeartIcon(true);
          }
        }}
      />
    );
  };

  return (
    <div
      suppressHydrationWarning
      className={classNames(
        // 使用固定宽度，避免非标准 w-59 导致不生效
        // 小屏默认折叠（w-0），点击按钮后在小屏展开到 200px；md 及以上始终 240px
        'flex flex-col bg-gray-900 text-gray-100 transition-all duration-300',
        isOpenOnMobile ? 'w-[200px] md:w-[240px]' : 'w-0 md:w-[240px] overflow-hidden'
      )}
    >
      <div className="px-4 md:px-3 py-3">
        {/* 桌面端将标题整体左移少量（md:px-3），避免右侧被遮挡 */}
        {/* 移动端用 grid 实现两行对齐：第二行与第一行文字对齐；桌面端保持原 flex 排版 */}
        <div className="grid grid-cols-[auto_1fr] items-center gap-3 md:flex md:items-center">
          {/* 左侧 Logo（移动端占两行，保证右侧两行文字纵向对齐） */}
          <img src="/ostris_logo.png" alt="Ostris AI Toolkit" className="w-auto h-7 md:mr-3 row-span-2 md:row-span-1" />

          {/* 第一行标题：OSTRIS */}
          <span className="font-bold uppercase md:ml-0">OSTRIS</span>

          {/* 第二行副标题：AI-TOOLKIT-E2U（不拆分，移动端与第一行文字左缘对齐） */}
          <div className="uppercase text-gray-300 text-sm whitespace-nowrap md:ml-0">AI-TOOLKIT-E2U</div>
        </div>
      </div>
      <nav className="flex-1">
        <ul className="px-2 py-4 space-y-2">
          {navigation.map(item => (
            <li key={item.name}>
              <Link
                href={item.href}
                className="flex items-center px-4 py-2 text-gray-300 hover:bg-gray-800 rounded-lg transition-colors"
                onClick={() => {
                  // 移动端点击后自动收起侧边栏
                  if (typeof window !== 'undefined' && window.innerWidth < 768) {
                    setIsOpenOnMobile(false);
                  }
                }}
              >
                <item.icon className="w-5 h-5 mr-3" />
                {item.name}
              </Link>
            </li>
          ))}
        </ul>
      </nav>
      <div className="flex items-center space-x-2 px-4 py-3">
        <div className="min-w-[26px] min-h-[26px]">
          {/* 头像加载失败时回退到爱心图标 */}
          <AvatarOrHeart />
        </div>
        <div className="text-gray-500 text-sm mb-2 flex-1 pt-2 pl-0">由Doc_workBox汉化</div>
      </div>

      {/* Social links grid */}
      <div className="px-1 py-1 border-t border-gray-800">
        <div className="grid grid-cols-2 gap-4">
          <a href="https://www.youtube.com/@Doc_workBox" target="_blank" rel="noreferrer" className={socialsBoxClass}>
            <FaYoutube className={socialIconClass} />
          </a>
          <a href="https://space.bilibili.com/12710942" target="_blank" rel="noreferrer" className={socialsBoxClass}>
            <SiBilibili className={socialIconClass} />
          </a>
        </div>
      </div>

      {/* 按钮已移至 TopBar，通过全局事件控制。保留此处空段以便未来扩展。 */}
    </div>
  );
};

export default Sidebar;
