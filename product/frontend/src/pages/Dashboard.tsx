import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Users, Eye, MessageSquare, FileText, Plus, ArrowRight } from 'lucide-react';
import { arquetiposApi, analisisApi, interaccionApi, reportesApi } from '../lib/api';

interface Stats {
  arquetipos: number;
  analisis: number;
  sesiones: number;
  reportes: number;
}

export function Dashboard() {
  const [stats, setStats] = useState<Stats>({ arquetipos: 0, analisis: 0, sesiones: 0, reportes: 0 });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadStats() {
      try {
        const [arq, ana, ses, rep] = await Promise.all([
          arquetiposApi.list(),
          analisisApi.list(),
          interaccionApi.listSessions(),
          reportesApi.list(),
        ]);
        setStats({
          arquetipos: arq.total,
          analisis: ana.total,
          sesiones: ses.total,
          reportes: rep.total,
        });
      } catch (error) {
        console.error('Error loading stats:', error);
      } finally {
        setLoading(false);
      }
    }
    loadStats();
  }, []);

  const cards = [
    {
      title: 'Arquetipos',
      count: stats.arquetipos,
      icon: Users,
      color: 'bg-primary',
      link: '/arquetipos',
      action: 'Crear arquetipo',
    },
    {
      title: 'Analisis Visual',
      count: stats.analisis,
      icon: Eye,
      color: 'bg-secondary',
      link: '/analisis',
      action: 'Nuevo analisis',
    },
    {
      title: 'Sesiones',
      count: stats.sesiones,
      icon: MessageSquare,
      color: 'bg-accent',
      link: '/interaccion',
      action: 'Iniciar chat',
    },
    {
      title: 'Reportes',
      count: stats.reportes,
      icon: FileText,
      color: 'bg-success',
      link: '/reportes',
      action: 'Generar reporte',
    },
  ];

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-text-primary">Dashboard</h1>
        <p className="text-text-secondary mt-1">
          Bienvenido a Usuarios Sinteticos
        </p>
      </div>

      {/* Stats cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {cards.map(({ title, count, icon: Icon, color, link, action }) => (
          <div
            key={title}
            className="bg-bg-secondary rounded-xl p-6 border border-border hover:border-border-dark transition-colors"
          >
            <div className="flex items-start justify-between mb-4">
              <div className={`p-3 rounded-lg ${color}`}>
                <Icon size={24} className="text-white" />
              </div>
              {loading ? (
                <div className="w-12 h-8 bg-bg-tertiary rounded animate-pulse" />
              ) : (
                <span className="text-3xl font-bold text-text-primary">{count}</span>
              )}
            </div>
            <h3 className="font-medium text-text-primary mb-3">{title}</h3>
            <Link
              to={link}
              className="inline-flex items-center gap-1.5 text-sm text-primary hover:text-primary-dark transition-colors"
            >
              <Plus size={16} />
              {action}
            </Link>
          </div>
        ))}
      </div>

      {/* Quick actions */}
      <div className="bg-bg-secondary rounded-xl p-6 border border-border">
        <h2 className="text-lg font-semibold text-text-primary mb-4">Acciones rapidas</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Link
            to="/arquetipos/nuevo"
            className="flex items-center justify-between p-4 rounded-lg border border-border hover:border-primary hover:bg-primary/5 transition-colors group"
          >
            <div className="flex items-center gap-3">
              <Users size={20} className="text-primary" />
              <span className="font-medium text-text-primary">Crear arquetipo</span>
            </div>
            <ArrowRight size={18} className="text-text-muted group-hover:text-primary transition-colors" />
          </Link>

          <Link
            to="/analisis/nuevo"
            className="flex items-center justify-between p-4 rounded-lg border border-border hover:border-secondary hover:bg-secondary/5 transition-colors group"
          >
            <div className="flex items-center gap-3">
              <Eye size={20} className="text-secondary" />
              <span className="font-medium text-text-primary">Analizar diseno</span>
            </div>
            <ArrowRight size={18} className="text-text-muted group-hover:text-secondary transition-colors" />
          </Link>

          <Link
            to="/interaccion"
            className="flex items-center justify-between p-4 rounded-lg border border-border hover:border-accent hover:bg-accent/5 transition-colors group"
          >
            <div className="flex items-center gap-3">
              <MessageSquare size={20} className="text-accent" />
              <span className="font-medium text-text-primary">Chat sintetico</span>
            </div>
            <ArrowRight size={18} className="text-text-muted group-hover:text-accent transition-colors" />
          </Link>
        </div>
      </div>
    </div>
  );
}
