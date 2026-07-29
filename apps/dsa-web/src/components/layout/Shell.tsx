import type React from 'react';
import { useCallback, useEffect, useState } from 'react';
import { Menu, PanelLeftClose, PanelLeftOpen } from 'lucide-react';
import { Outlet } from 'react-router-dom';
import { Drawer } from '../common/Drawer';
import { SidebarNav } from './SidebarNav';
import { cn } from '../../utils/cn';
import { ThemeToggle } from '../theme/ThemeToggle';
import { UiLanguageToggle } from '../i18n/UiLanguageToggle';
import { useUiLanguage } from '../../contexts/UiLanguageContext';

const SIDEBAR_COLLAPSED_KEY = 'sidebar-collapsed';

type ShellProps = {
  children?: React.ReactNode;
};

export const Shell: React.FC<ShellProps> = ({ children }) => {
  const [mobileOpen, setMobileOpen] = useState(false);
  const [collapsed, setCollapsed] = useState(() => {
    try {
      return localStorage.getItem(SIDEBAR_COLLAPSED_KEY) === 'true';
    } catch {
      return false;
    }
  });
  const { t } = useUiLanguage();

  const toggleCollapsed = useCallback(() => {
    setCollapsed((prev) => {
      const next = !prev;
      try {
        localStorage.setItem(SIDEBAR_COLLAPSED_KEY, String(next));
      } catch { /* ignore */ }
      return next;
    });
  }, []);

  useEffect(() => {
    if (!mobileOpen) {
      return undefined;
    }

    const handleResize = () => {
      if (window.innerWidth >= 1024) {
        setMobileOpen(false);
      }
    };

    window.addEventListener('resize', handleResize);
    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, [mobileOpen]);

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="pointer-events-none fixed inset-x-0 top-3 z-40 flex items-start justify-between px-3 lg:hidden">
        <button
          type="button"
          onClick={() => setMobileOpen(true)}
          className="pointer-events-auto inline-flex h-10 w-10 items-center justify-center rounded-xl border border-border/70 bg-card/85 text-secondary-text shadow-soft-card backdrop-blur-md transition-colors hover:bg-hover hover:text-foreground"
          aria-label={t('layout.openNav')}
        >
          <Menu className="h-5 w-5" />
        </button>
        <div className="pointer-events-auto flex items-center gap-2">
          <UiLanguageToggle />
          <ThemeToggle />
        </div>
      </div>

      <div className="mx-auto flex min-h-screen w-full max-w-[1680px] px-3 py-3 sm:px-4 sm:py-4 lg:px-5">
        <aside
          className={cn(
            'sticky top-3 z-40 hidden shrink-0 flex-col overflow-visible rounded-[1.5rem] border border-[var(--shell-sidebar-border)] bg-card/72 p-2.5 shadow-soft-card backdrop-blur-sm transition-[width] duration-200 lg:flex',
            'max-h-[calc(100vh-1.5rem)] self-start sm:top-4 sm:max-h-[calc(100vh-2rem)]',
            collapsed ? 'w-[64px]' : 'w-[168px]'
          )}
          aria-label={t('layout.desktopSidebar')}
        >
          <SidebarNav collapsed={collapsed} variant="rail" onNavigate={() => setMobileOpen(false)} />
          <button
            type="button"
            onClick={toggleCollapsed}
            className="mt-1.5 flex h-10 w-full items-center justify-center rounded-2xl border border-transparent text-secondary-text transition-colors hover:bg-[var(--nav-hover-bg)] hover:text-foreground"
            aria-label={collapsed ? t('layout.expandSidebar') : t('layout.collapseSidebar')}
          >
            {collapsed ? <PanelLeftOpen className="h-[18px] w-[18px]" /> : <PanelLeftClose className="h-[18px] w-[18px]" />}
          </button>
        </aside>

        <main className="min-h-0 min-w-0 flex-1 pt-14 lg:pl-3 lg:pt-0 touch-pan-y">
          {children ?? <Outlet />}
        </main>
      </div>

      <Drawer
        isOpen={mobileOpen}
        onClose={() => setMobileOpen(false)}
        title={t('layout.navMenu')}
        width="max-w-xs"
        zIndex={90}
        side="left"
      >
        <SidebarNav onNavigate={() => setMobileOpen(false)} />
      </Drawer>
    </div>
  );
};
