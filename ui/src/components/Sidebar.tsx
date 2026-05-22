'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Home, Settings, BrainCircuit, Images, Plus, X, Heart } from 'lucide-react';
import { FaYoutube } from 'react-icons/fa6';
import { SiBilibili } from 'react-icons/si';
import { createGlobalState } from 'react-global-hooks';
import ThemeToggle from './ThemeToggle';
import ThemeLogo from './ThemeLogo';

export const mobileSidebarState = createGlobalState<boolean>(false);

const Sidebar = () => {
  const [isMobileOpen, setIsMobileOpen] = mobileSidebarState.use();
  const pathname = usePathname();

  useEffect(() => {
    setIsMobileOpen(false);
  }, [pathname, setIsMobileOpen]);

  useEffect(() => {
    document.body.style.overflow = isMobileOpen ? 'hidden' : '';
    return () => {
      document.body.style.overflow = '';
    };
  }, [isMobileOpen]);

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

  const AvatarOrHeart = () => {
    const [useHeartIcon, setUseHeartIcon] = useState(false);
    const [srcIndex, setSrcIndex] = useState(0);
    const candidates = [
      '/doc_workbox_avatar.jpg',
      '/doc_workbox_avatar.png',
      '/doc_workbox_avatar.jpeg',
      '/doc_workbox_avatar.webp',
      '/doc_workbox_avatar.svg',
      '/doc_workbox_avatar.avif',
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

  const sidebarContent = (
    <>
      <div className="px-4 py-3 flex items-center justify-between">
        <h1 className="text-l flex items-center gap-2">
          <ThemeLogo />
          <span className="font-bold uppercase">OSTRIS</span>
          <span className="uppercase text-gray-300">AI-TOOLKIT-E2U</span>
        </h1>
        <button
          onClick={() => setIsMobileOpen(false)}
          className="md:hidden text-gray-400 hover:text-white p-1"
          aria-label="关闭菜单"
        >
          <X className="w-5 h-5" />
        </button>
      </div>
      <nav className="flex-1">
        <ul className="px-2 py-4 space-y-2">
          {navigation.map(item => (
            <li key={item.name}>
              <Link
                href={item.href}
                className="flex items-center px-4 py-2 text-gray-300 hover:bg-gray-800 rounded-lg transition-colors"
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
          <AvatarOrHeart />
        </div>
        <div className="text-gray-500 text-sm mb-2 flex-1 pt-2 pl-0">由Doc_workBox汉化</div>
      </div>

      <div className="px-1 py-1 border-t border-gray-800">
        <div className="grid grid-cols-3 gap-3">
          <a href="https://www.youtube.com/@Doc_workBox" target="_blank" rel="noreferrer" className={socialsBoxClass}>
            <FaYoutube className={socialIconClass} />
          </a>
          <a href="https://space.bilibili.com/12710942" target="_blank" rel="noreferrer" className={socialsBoxClass}>
            <SiBilibili className={socialIconClass} />
          </a>
          <ThemeToggle />
        </div>
      </div>
    </>
  );

  return (
    <>
      <div className="hidden md:flex flex-col w-59 bg-gray-900 text-gray-100">{sidebarContent}</div>
      <div
        className={`md:hidden fixed inset-0 bg-black/60 z-40 transition-opacity duration-300 ease-in-out ${
          isMobileOpen ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'
        }`}
        onClick={() => setIsMobileOpen(false)}
        aria-hidden="true"
      />
      <div
        className={`md:hidden fixed top-0 left-0 bottom-0 w-64 max-w-[85vw] bg-gray-900 text-gray-100 z-50 flex flex-col shadow-xl transform transition-transform duration-300 ease-in-out ${
          isMobileOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        {sidebarContent}
      </div>
    </>
  );
};

export default Sidebar;
