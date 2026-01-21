import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Plus, Eye, Image, ExternalLink } from 'lucide-react';
import { analisisApi, type Analisis } from '../lib/api';

export function AnalisisPage() {
  const [analisis, setAnalisis] = useState<Analisis[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadAnalisis();
  }, []);

  async function loadAnalisis() {
    try {
      const { analisis } = await analisisApi.list();
      setAnalisis(analisis);
    } catch (error) {
      console.error('Error loading analisis:', error);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-text-primary">Analisis Visual</h1>
          <p className="text-text-secondary mt-1">Analiza tus disenos con Vision AI</p>
        </div>
        <Link
          to="/analisis/nuevo"
          className="flex items-center gap-2 px-4 py-2 bg-secondary hover:bg-secondary/90 text-white rounded-lg transition-colors"
        >
          <Plus size={18} />
          Nuevo analisis
        </Link>
      </div>

      {/* List */}
      {loading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {[1, 2, 3].map(i => (
            <div key={i} className="bg-bg-secondary rounded-xl overflow-hidden animate-pulse">
              <div className="h-40 bg-bg-tertiary" />
              <div className="p-4">
                <div className="h-4 bg-bg-tertiary rounded w-1/2 mb-2" />
                <div className="h-3 bg-bg-tertiary rounded w-3/4" />
              </div>
            </div>
          ))}
        </div>
      ) : analisis.length === 0 ? (
        <div className="text-center py-12 bg-bg-secondary rounded-xl border border-border">
          <Eye size={48} className="mx-auto text-text-muted mb-4" />
          <h3 className="text-lg font-medium text-text-primary mb-2">No hay analisis</h3>
          <p className="text-text-secondary mb-4">Sube una imagen o URL para comenzar</p>
          <Link
            to="/analisis/nuevo"
            className="inline-flex items-center gap-2 px-4 py-2 bg-secondary hover:bg-secondary/90 text-white rounded-lg transition-colors"
          >
            <Plus size={18} />
            Nuevo analisis
          </Link>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {analisis.map(item => (
            <div
              key={item.id}
              className="bg-bg-secondary rounded-xl overflow-hidden border border-border hover:border-border-dark transition-colors"
            >
              <div className="h-40 bg-bg-tertiary relative">
                {item.imagen_url && !item.imagen_url.startsWith('upload://') ? (
                  <img
                    src={item.imagen_url}
                    alt="Analisis"
                    className="w-full h-full object-cover"
                  />
                ) : (
                  <div className="w-full h-full flex items-center justify-center">
                    <Image size={32} className="text-text-muted" />
                  </div>
                )}
                {item.clarity_score !== undefined && (
                  <div className="absolute top-2 right-2 px-2 py-1 bg-black/70 text-white text-xs rounded">
                    Clarity: {item.clarity_score}%
                  </div>
                )}
              </div>
              <div className="p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-text-muted">
                    {item.created_at ? new Date(item.created_at).toLocaleDateString() : 'Sin fecha'}
                  </span>
                  <div className="flex gap-1">
                    <Link
                      to={`/analisis/${item.id}`}
                      className="p-1.5 text-text-muted hover:text-secondary hover:bg-bg-tertiary rounded transition-colors"
                    >
                      <ExternalLink size={16} />
                    </Link>
                  </div>
                </div>
                {item.insights && item.insights.length > 0 && (
                  <p className="text-sm text-text-secondary line-clamp-2">
                    {item.insights[0]}
                  </p>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
