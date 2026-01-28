import { useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import {
  ArrowLeft,
  Loader2,
  Target,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  TrendingUp,
  TrendingDown,
  Users,
  MousePointer2,
} from 'lucide-react';
import {
  usabilityApi,
  flujosApi,
  type FlujoDetail,
  type Mision,
  type UsabilityMetricas,
} from '../lib/api';

interface MisionWithMetrics extends Mision {
  metricas?: UsabilityMetricas;
}

export function UsabilityDashboard() {
  const { flujoId } = useParams();
  const navigate = useNavigate();

  const [flujo, setFlujo] = useState<FlujoDetail | null>(null);
  const [misiones, setMisiones] = useState<MisionWithMetrics[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (flujoId) loadData();
  }, [flujoId]);

  async function loadData() {
    setLoading(true);
    try {
      const [flujoRes, misionesRes] = await Promise.all([
        flujosApi.get(flujoId!),
        usabilityApi.listMisiones(flujoId!),
      ]);

      setFlujo(flujoRes);

      // Load metrics for each mision
      const misionesWithMetrics = await Promise.all(
        misionesRes.misiones.map(async (mision) => {
          try {
            const metricas = await usabilityApi.getMetricas(mision.id);
            return { ...mision, metricas };
          } catch {
            return mision;
          }
        })
      );

      setMisiones(misionesWithMetrics);
    } catch (error) {
      console.error('Error loading data:', error);
    } finally {
      setLoading(false);
    }
  }

  // Calculate aggregated metrics
  const aggregatedMetrics = misiones.reduce(
    (acc, mision) => {
      if (mision.metricas && mision.metricas.total_simulaciones > 0) {
        acc.totalSimulaciones += mision.metricas.total_simulaciones;
        acc.totalExitosas += Math.round(
          (mision.metricas.success_rate / 100) * mision.metricas.total_simulaciones
        );
        acc.totalMisclicks += mision.metricas.avg_misclicks * mision.metricas.total_simulaciones;
        acc.totalFricciones += mision.metricas.total_fricciones;
      }
      return acc;
    },
    { totalSimulaciones: 0, totalExitosas: 0, totalMisclicks: 0, totalFricciones: 0 }
  );

  const overallSuccessRate =
    aggregatedMetrics.totalSimulaciones > 0
      ? (aggregatedMetrics.totalExitosas / aggregatedMetrics.totalSimulaciones) * 100
      : 0;

  const overallMisclicks =
    aggregatedMetrics.totalSimulaciones > 0
      ? aggregatedMetrics.totalMisclicks / aggregatedMetrics.totalSimulaciones
      : 0;

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <Loader2 size={32} className="animate-spin text-primary" />
      </div>
    );
  }

  if (!flujo) {
    return (
      <div className="text-center py-12">
        <p className="text-text-secondary">Flujo no encontrado</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-4">
        <button
          onClick={() => navigate(`/flujos/${flujoId}`)}
          className="p-2 text-text-secondary hover:text-text-primary hover:bg-bg-tertiary rounded-lg transition-colors"
        >
          <ArrowLeft size={20} />
        </button>
        <div>
          <h1 className="text-xl font-bold text-text-primary">Dashboard de Usability</h1>
          <p className="text-sm text-text-secondary">{flujo.nombre}</p>
        </div>
      </div>

      {/* Overall Metrics */}
      {aggregatedMetrics.totalSimulaciones > 0 && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="p-4 bg-bg-secondary rounded-xl border border-border">
            <div className="flex items-center justify-between mb-2">
              <Users size={20} className="text-primary" />
              <span className="text-2xl font-bold text-text-primary">
                {aggregatedMetrics.totalSimulaciones}
              </span>
            </div>
            <p className="text-sm text-text-muted">Total Simulaciones</p>
          </div>

          <div
            className={`p-4 rounded-xl border ${
              overallSuccessRate >= 70
                ? 'bg-success/5 border-success/20'
                : overallSuccessRate >= 40
                ? 'bg-warning/5 border-warning/20'
                : 'bg-error/5 border-error/20'
            }`}
          >
            <div className="flex items-center justify-between mb-2">
              {overallSuccessRate >= 70 ? (
                <TrendingUp size={20} className="text-success" />
              ) : (
                <TrendingDown size={20} className="text-error" />
              )}
              <span
                className={`text-2xl font-bold ${
                  overallSuccessRate >= 70
                    ? 'text-success'
                    : overallSuccessRate >= 40
                    ? 'text-warning'
                    : 'text-error'
                }`}
              >
                {Math.round(overallSuccessRate)}%
              </span>
            </div>
            <p className="text-sm text-text-muted">Success Rate</p>
          </div>

          <div
            className={`p-4 rounded-xl border ${
              overallMisclicks <= 1
                ? 'bg-success/5 border-success/20'
                : overallMisclicks <= 2
                ? 'bg-warning/5 border-warning/20'
                : 'bg-error/5 border-error/20'
            }`}
          >
            <div className="flex items-center justify-between mb-2">
              <MousePointer2 size={20} className="text-warning" />
              <span className="text-2xl font-bold text-text-primary">
                {overallMisclicks.toFixed(1)}
              </span>
            </div>
            <p className="text-sm text-text-muted">Avg Misclicks</p>
          </div>

          <div className="p-4 bg-bg-secondary rounded-xl border border-border">
            <div className="flex items-center justify-between mb-2">
              <AlertTriangle size={20} className="text-warning" />
              <span className="text-2xl font-bold text-text-primary">
                {aggregatedMetrics.totalFricciones}
              </span>
            </div>
            <p className="text-sm text-text-muted">Fricciones Detectadas</p>
          </div>
        </div>
      )}

      {/* Misiones Overview */}
      <div className="bg-bg-secondary rounded-xl border border-border overflow-hidden">
        <div className="p-4 border-b border-border">
          <h2 className="font-medium text-text-primary">Resumen de Misiones</h2>
        </div>

        {misiones.length === 0 ? (
          <div className="p-8 text-center text-text-muted">
            <Target size={48} className="mx-auto mb-3 opacity-50" />
            <p>No hay misiones de usability testing</p>
            <button
              onClick={() => navigate(`/flujos/${flujoId}`)}
              className="mt-4 text-primary hover:underline text-sm"
            >
              Crear primera mision
            </button>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-bg-tertiary">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-medium text-text-muted uppercase tracking-wide">
                    Mision
                  </th>
                  <th className="px-4 py-3 text-center text-xs font-medium text-text-muted uppercase tracking-wide">
                    Simulaciones
                  </th>
                  <th className="px-4 py-3 text-center text-xs font-medium text-text-muted uppercase tracking-wide">
                    Success Rate
                  </th>
                  <th className="px-4 py-3 text-center text-xs font-medium text-text-muted uppercase tracking-wide">
                    Misclicks
                  </th>
                  <th className="px-4 py-3 text-center text-xs font-medium text-text-muted uppercase tracking-wide">
                    Fricciones
                  </th>
                  <th className="px-4 py-3 text-center text-xs font-medium text-text-muted uppercase tracking-wide">
                    Estado
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {misiones.map((mision) => {
                  const hasMetrics =
                    mision.metricas && mision.metricas.total_simulaciones > 0;
                  const successRate = hasMetrics ? mision.metricas!.success_rate : 0;

                  return (
                    <tr
                      key={mision.id}
                      onClick={() => navigate(`/flujos/${flujoId}/misiones/${mision.id}`)}
                      className="hover:bg-bg-tertiary cursor-pointer"
                    >
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-2">
                          <Target size={14} className="text-primary" />
                          <div>
                            <p className="font-medium text-text-primary text-sm">
                              {mision.nombre}
                            </p>
                            <p className="text-xs text-text-muted truncate max-w-xs">
                              {mision.instrucciones}
                            </p>
                          </div>
                        </div>
                      </td>
                      <td className="px-4 py-3 text-center">
                        <span className="text-sm text-text-primary">
                          {hasMetrics ? mision.metricas!.total_simulaciones : 0}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-center">
                        {hasMetrics ? (
                          <span
                            className={`inline-flex items-center gap-1 text-sm font-medium ${
                              successRate >= 70
                                ? 'text-success'
                                : successRate >= 40
                                ? 'text-warning'
                                : 'text-error'
                            }`}
                          >
                            {successRate >= 70 ? (
                              <CheckCircle2 size={14} />
                            ) : (
                              <XCircle size={14} />
                            )}
                            {Math.round(successRate)}%
                          </span>
                        ) : (
                          <span className="text-text-muted">-</span>
                        )}
                      </td>
                      <td className="px-4 py-3 text-center">
                        <span className="text-sm text-text-primary">
                          {hasMetrics ? mision.metricas!.avg_misclicks.toFixed(1) : '-'}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-center">
                        <span className="text-sm text-text-primary">
                          {hasMetrics ? mision.metricas!.total_fricciones : '-'}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-center">
                        <span
                          className={`text-xs px-2 py-0.5 rounded ${
                            mision.estado === 'activa'
                              ? 'bg-success/10 text-success'
                              : 'bg-text-muted/10 text-text-muted'
                          }`}
                        >
                          {mision.estado}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Recommendations */}
      {aggregatedMetrics.totalSimulaciones > 0 && (
        <div className="bg-bg-secondary rounded-xl border border-border p-4">
          <h3 className="font-medium text-text-primary mb-3">Recomendaciones</h3>
          <div className="space-y-2">
            {overallSuccessRate < 70 && (
              <div className="flex items-start gap-2 p-3 bg-warning/5 border border-warning/20 rounded-lg">
                <AlertTriangle size={16} className="text-warning mt-0.5" />
                <div>
                  <p className="text-sm font-medium text-warning">Success rate bajo</p>
                  <p className="text-xs text-text-muted mt-0.5">
                    El success rate esta por debajo del 70%. Considera revisar la claridad de los
                    CTAs y la jerarquia visual.
                  </p>
                </div>
              </div>
            )}

            {overallMisclicks > 2 && (
              <div className="flex items-start gap-2 p-3 bg-error/5 border border-error/20 rounded-lg">
                <MousePointer2 size={16} className="text-error mt-0.5" />
                <div>
                  <p className="text-sm font-medium text-error">Alto numero de misclicks</p>
                  <p className="text-xs text-text-muted mt-0.5">
                    Los usuarios cometen muchos clicks erroneos. Revisa el tamano de los elementos
                    y la separacion entre ellos.
                  </p>
                </div>
              </div>
            )}

            {aggregatedMetrics.totalFricciones > misiones.length * 2 && (
              <div className="flex items-start gap-2 p-3 bg-orange-500/5 border border-orange-500/20 rounded-lg">
                <AlertTriangle size={16} className="text-orange-500 mt-0.5" />
                <div>
                  <p className="text-sm font-medium text-orange-500">
                    Multiples puntos de friccion
                  </p>
                  <p className="text-xs text-text-muted mt-0.5">
                    Se detectaron varios puntos de friccion. Revisa los detalles de cada simulacion
                    para identificar patrones.
                  </p>
                </div>
              </div>
            )}

            {overallSuccessRate >= 70 && overallMisclicks <= 1 && (
              <div className="flex items-start gap-2 p-3 bg-success/5 border border-success/20 rounded-lg">
                <CheckCircle2 size={16} className="text-success mt-0.5" />
                <div>
                  <p className="text-sm font-medium text-success">Buen rendimiento</p>
                  <p className="text-xs text-text-muted mt-0.5">
                    Los usuarios sinteticos navegan bien el flujo. Considera probar con mas
                    arquetipos para validar.
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
