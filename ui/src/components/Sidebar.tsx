"use client";
import Link from 'next/link';
import { Home, Settings, BrainCircuit, Images, Plus, Heart } from 'lucide-react';
import { FaYoutube } from 'react-icons/fa6';
import { SiBilibili } from 'react-icons/si';
import { useState } from 'react';

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
    <div className="flex flex-col w-59 bg-gray-900 text-gray-100">
      <div className="px-4 py-3">
        <h1 className="text-l">
          <img src="/ostris_logo.png" alt="Ostris AI Toolkit" className="w-auto h-7 mr-3 inline" />
          <span className="font-bold uppercase">Ostris</span>
          <span className="ml-2 uppercase text-gray-300">AI-Toolkit</span>
        </h1>
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
    </div>
  );
};

export default Sidebar;
