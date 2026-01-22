import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  Users,
  Eye,
  MessageSquare,
  FileText,
  LogOut,
  ChevronLeft,
  ChevronRight,
  GitCompare
} from 'lucide-react';
import { useState } from 'react';
import { useAuth } from '../../context/AuthContext';
import clsx from 'clsx';

const navItems = [
  { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/arquetipos', icon: Users, label: 'Arquetipos' },
  { to: '/analisis', icon: Eye, label: 'Analisis Visual' },
  { to: '/analisis/comparar-modelos', icon: GitCompare, label: 'Comparar Modelos' },
  { to: '/interaccion', icon: MessageSquare, label: 'Interaccion' },
  { to: '/reportes', icon: FileText, label: 'Reportes' },
];

export function Sidebar() {
  const [collapsed, setCollapsed] = useState(false);
  const { signOut, user } = useAuth();

  return (
    <aside
      className={clsx(
        'flex flex-col h-screen bg-bg-secondary border-r border-border transition-all duration-300',
        collapsed ? 'w-16' : 'w-64'
      )}
    >
      {/* Logo */}
      <div className="flex items-center h-16 px-4 border-b border-border">
        {!collapsed && (
          <span className="text-lg font-semibold text-primary">
            Usuarios Sinteticos
          </span>
        )}
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="ml-auto p-1.5 rounded-lg hover:bg-bg-tertiary text-text-secondary"
        >
          {collapsed ? <ChevronRight size={20} /> : <ChevronLeft size={20} />}
        </button>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-2 py-4 space-y-1">
        {navItems.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              clsx(
                'flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors',
                isActive
                  ? 'bg-primary text-white'
                  : 'text-text-secondary hover:bg-bg-tertiary hover:text-text-primary'
              )
            }
          >
            <Icon size={20} />
            {!collapsed && <span>{label}</span>}
          </NavLink>
        ))}
      </nav>

      {/* User section */}
      <div className="p-4 border-t border-border">
        {!collapsed && user && (
          <p className="text-sm text-text-muted truncate mb-2">{user.email}</p>
        )}
        <button
          onClick={signOut}
          className={clsx(
            'flex items-center gap-3 w-full px-3 py-2 rounded-lg text-text-secondary hover:bg-bg-tertiary hover:text-error transition-colors',
            collapsed && 'justify-center'
          )}
        >
          <LogOut size={20} />
          {!collapsed && <span>Cerrar sesion</span>}
        </button>
      </div>
    </aside>
  );
}
