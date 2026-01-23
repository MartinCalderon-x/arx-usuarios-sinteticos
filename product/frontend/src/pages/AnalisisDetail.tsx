import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft, Download, Eye, Layers, Target, Lightbulb, Clock, Cpu } from 'lucide-react';
import { analisisApi, type Analisis } from '../lib/api';

type ViewMode = 'original' | 'heatmap' | 'overlay';

export function AnalisisDetail() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [analisis, setAnalisis] = useState<Analisis | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>('overlay');

  useEffect(() => {
    if (id) {
      loadAnalisis(id);
    }
  }, [id]);

  async function loadAnalisis(analisisId: string) {
    try {
      setLoading(true);
      const data = await analisisApi.get(analisisId);
      setAnalisis(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Error al cargar analisis');
    } finally {
      setLoading(false);
    }
  }

  function getCurrentImageUrl(): string | null {
    if (!analisis) return null;
    switch (viewMode) {
      case 'original':
        return analisis.imagen_url;
      case 'heatmap':
        return analisis.heatmap_url || null;
      case 'overlay':
        return analisis.focus_map_url || analisis.heatmap_url || null;
      default:
        return analisis.imagen_url;
    }
  }

  function getScoreColor(score: number): string {
    if (score >= 70) return 'text-green-500';
    if (score >= 40) return 'text-yellow-500';
    return 'text-red-500';
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="w-8 h-8 border-2 border-primary/30 border-t-primary rounded-full animate-spin" />
      </div>
    );
  }

  if (error || !analisis) {
    return (
      <div className="text-center py-12">
        <p className="text-red-500 mb-4">{error || 'Analisis no encontrado'}</p>
        <button
          onClick={() => navigate('/analisis')}
          className="text-primary hover:underline"
        >
          Volver a la lista
        </button>
      </div>
    );
  }

  const currentImage = getCurrentImageUrl();

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-4">
        <button
          onClick={() => navigate('/analisis')}
          className="p-2 rounded-lg hover:bg-bg-secondary text-text-secondary transition-colors"
        >
          <ArrowLeft size={20} />
        </button>
        <div className="flex-1">
          <h1 className="text-2xl font-bold text-text-primary">Resultado del Analisis</h1>
          <p className="text-text-secondary mt-1">
            {analisis.created_at && new Date(analisis.created_at).toLocaleString()}
          </p>
        </div>
        {analisis.modelo_usado && (
          <div className="flex items-center gap-2 px-3 py-1.5 bg-bg-secondary rounded-lg text-sm">
            <Cpu size={16} className="text-primary" />
            <span className="text-text-secondary">{analisis.modelo_usado}</span>
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main Image Viewer */}
        <div className="lg:col-span-2 space-y-4">
          {/* View Mode Selector */}
          <div className="flex gap-2 p-1 bg-bg-secondary rounded-lg">
            <button
              onClick={() => setViewMode('original')}
              className={`flex-1 flex items-center justify-center gap-2 px-4 py-2 rounded-md transition-colors ${
                viewMode === 'original'
                  ? 'bg-bg-primary shadow text-primary'
                  : 'text-text-secondary hover:text-text-primary'
              }`}
            >
              <Eye size={18} />
              Original
            </button>
            <button
              onClick={() => setViewMode('heatmap')}
              disabled={!analisis.heatmap_url}
              className={`flex-1 flex items-center justify-center gap-2 px-4 py-2 rounded-md transition-colors ${
                viewMode === 'heatmap'
                  ? 'bg-bg-primary shadow text-primary'
                  : 'text-text-secondary hover:text-text-primary'
              } disabled:opacity-50 disabled:cursor-not-allowed`}
            >
              <Target size={18} />
              Heatmap
            </button>
            <button
              onClick={() => setViewMode('overlay')}
              disabled={!analisis.focus_map_url && !analisis.heatmap_url}
              className={`flex-1 flex items-center justify-center gap-2 px-4 py-2 rounded-md transition-colors ${
                viewMode === 'overlay'
                  ? 'bg-bg-primary shadow text-primary'
                  : 'text-text-secondary hover:text-text-primary'
              } disabled:opacity-50 disabled:cursor-not-allowed`}
            >
              <Layers size={18} />
              Overlay
            </button>
          </div>

          {/* Image Display */}
          <div className="bg-bg-secondary rounded-xl overflow-hidden border border-border">
            {currentImage ? (
              <img
                src={currentImage}
                alt={`Vista ${viewMode}`}
                className="w-full h-auto max-h-[600px] object-contain bg-bg-tertiary"
              />
            ) : (
              <div className="flex items-center justify-center h-64 bg-bg-tertiary">
                <p className="text-text-muted">Imagen no disponible</p>
              </div>
            )}
          </div>

          {/* Download buttons */}
          <div className="flex gap-2">
            {analisis.imagen_url && (
              <a
                href={analisis.imagen_url}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 px-3 py-2 bg-bg-secondary hover:bg-bg-tertiary text-text-secondary rounded-lg transition-colors text-sm"
              >
                <Download size={16} />
                Original
              </a>
            )}
            {analisis.heatmap_url && (
              <a
                href={analisis.heatmap_url}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 px-3 py-2 bg-bg-secondary hover:bg-bg-tertiary text-text-secondary rounded-lg transition-colors text-sm"
              >
                <Download size={16} />
                Heatmap
              </a>
            )}
            {analisis.focus_map_url && (
              <a
                href={analisis.focus_map_url}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 px-3 py-2 bg-bg-secondary hover:bg-bg-tertiary text-text-secondary rounded-lg transition-colors text-sm"
              >
                <Download size={16} />
                Overlay
              </a>
            )}
          </div>
        </div>

        {/* Sidebar with metrics */}
        <div className="space-y-4">
          {/* Clarity Score */}
          {analisis.clarity_score !== undefined && (
            <div className="bg-bg-secondary rounded-xl p-4 border border-border">
              <h3 className="text-sm font-medium text-text-secondary mb-3 flex items-center gap-2">
                <Target size={16} />
                Clarity Score
              </h3>
              <div className="flex items-end gap-2">
                <span className={`text-4xl font-bold ${getScoreColor(analisis.clarity_score)}`}>
                  {analisis.clarity_score.toFixed(1)}
                </span>
                <span className="text-text-muted mb-1">/ 100</span>
              </div>
              <div className="mt-3 h-2 bg-bg-tertiary rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all ${
                    analisis.clarity_score >= 70
                      ? 'bg-green-500'
                      : analisis.clarity_score >= 40
                      ? 'bg-yellow-500'
                      : 'bg-red-500'
                  }`}
                  style={{ width: `${analisis.clarity_score}%` }}
                />
              </div>
            </div>
          )}

          {/* Areas de Interes */}
          {analisis.areas_interes && analisis.areas_interes.length > 0 && (
            <div className="bg-bg-secondary rounded-xl p-4 border border-border">
              <h3 className="text-sm font-medium text-text-secondary mb-3 flex items-center gap-2">
                <Eye size={16} />
                Areas de Interes
              </h3>
              <div className="space-y-2">
                {analisis.areas_interes
                  .sort((a, b) => a.orden_visual - b.orden_visual)
                  .map((area, index) => (
                    <div
                      key={index}
                      className="flex items-center gap-3 p-2 bg-bg-tertiary rounded-lg"
                    >
                      <span className="w-6 h-6 flex items-center justify-center bg-primary text-white text-xs font-bold rounded-full">
                        {area.orden_visual}
                      </span>
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium text-text-primary truncate">
                          {area.nombre}
                        </p>
                        <p className="text-xs text-text-muted">
                          Intensidad: {area.intensidad}%
                        </p>
                      </div>
                    </div>
                  ))}
              </div>
            </div>
          )}

          {/* Insights */}
          {analisis.insights && analisis.insights.length > 0 && (
            <div className="bg-bg-secondary rounded-xl p-4 border border-border">
              <h3 className="text-sm font-medium text-text-secondary mb-3 flex items-center gap-2">
                <Lightbulb size={16} />
                Insights
              </h3>
              <ul className="space-y-2">
                {analisis.insights.map((insight, index) => (
                  <li
                    key={index}
                    className="text-sm text-text-primary pl-4 border-l-2 border-primary/30"
                  >
                    {insight}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Metadata */}
          <div className="bg-bg-secondary rounded-xl p-4 border border-border">
            <h3 className="text-sm font-medium text-text-secondary mb-3 flex items-center gap-2">
              <Clock size={16} />
              Detalles
            </h3>
            <dl className="space-y-2 text-sm">
              <div className="flex justify-between">
                <dt className="text-text-muted">ID</dt>
                <dd className="text-text-primary font-mono text-xs">{analisis.id.slice(0, 8)}...</dd>
              </div>
              {analisis.modelo_usado && (
                <div className="flex justify-between">
                  <dt className="text-text-muted">Modelo</dt>
                  <dd className="text-text-primary">{analisis.modelo_usado}</dd>
                </div>
              )}
              {analisis.created_at && (
                <div className="flex justify-between">
                  <dt className="text-text-muted">Fecha</dt>
                  <dd className="text-text-primary">
                    {new Date(analisis.created_at).toLocaleDateString()}
                  </dd>
                </div>
              )}
            </dl>
          </div>
        </div>
      </div>
    </div>
  );
}
