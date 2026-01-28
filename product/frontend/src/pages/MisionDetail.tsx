import { useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import {
  ArrowLeft,
  Target,
  Loader2,
  Play,
  BarChart3,
  Users,
  CheckCircle2,
  XCircle,
  AlertTriangle,
} from 'lucide-react';
import {
  usabilityApi,
  flujosApi,
  arquetiposApi,
  type MisionDetail as MisionDetailType,
  type Simulacion,
  type UsabilityMetricas,
  type FlujoDetail,
  type Arquetipo,
} from '../lib/api';
import { SimulationPlayer } from '../components/flujos/SimulationPlayer';

export function MisionDetail() {
  const { flujoId, misionId } = useParams();
  const navigate = useNavigate();

  const [mision, setMision] = useState<MisionDetailType | null>(null);
  const [flujo, setFlujo] = useState<FlujoDetail | null>(null);
  const [metricas, setMetricas] = useState<UsabilityMetricas | null>(null);
  const [arquetipos, setArquetipos] = useState<Arquetipo[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedSimulacion, setSelectedSimulacion] = useState<Simulacion | null>(null);
  const [pantallaImagenes, setPantallaImagenes] = useState<Record<string, string>>({});
  const [runningSimulation, setRunningSimulation] = useState(false);
  const [selectedArquetipos, setSelectedArquetipos] = useState<string[]>([]);

  useEffect(() => {
    if (flujoId && misionId) loadData();
  }, [flujoId, misionId]);

  async function loadData() {
    setLoading(true);
    try {
      const [misionRes, flujoRes, metricasRes, arquetiposRes] = await Promise.all([
        usabilityApi.getMision(flujoId!, misionId!),
        flujosApi.get(flujoId!),
        usabilityApi.getMetricas(misionId!),
        arquetiposApi.list(),
      ]);

      setMision(misionRes);
      setFlujo(flujoRes);
      setMetricas(metricasRes);
      setArquetipos(arquetiposRes.arquetipos);

      // Build pantalla images map
      const imagesMap: Record<string, string> = {};
      for (const pantalla of flujoRes.pantallas) {
        if (pantalla.screenshot_url) {
          imagesMap[pantalla.id] = pantalla.screenshot_url;
        }
      }
      setPantallaImagenes(imagesMap);

      // Select first simulation by default
      if (misionRes.simulaciones.length > 0 && !selectedSimulacion) {
        setSelectedSimulacion(misionRes.simulaciones[0]);
      }
    } catch (error) {
      console.error('Error loading data:', error);
      navigate(`/flujos/${flujoId}`);
    } finally {
      setLoading(false);
    }
  }

  async function handleRunSimulation() {
    if (!misionId || selectedArquetipos.length === 0) return;

    setRunningSimulation(true);
    try {
      await usabilityApi.runSimulation(misionId, selectedArquetipos);
      setSelectedArquetipos([]);
      await loadData();
    } catch (error) {
      console.error('Error running simulation:', error);
      alert('Error al ejecutar simulacion');
    } finally {
      setRunningSimulation(false);
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <Loader2 size={32} className="animate-spin text-primary" />
      </div>
    );
  }

  if (!mision || !flujo) {
    return (
      <div className="text-center py-12">
        <p className="text-text-secondary">Mision no encontrada</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <button
            onClick={() => navigate(`/flujos/${flujoId}`)}
            className="p-2 text-text-secondary hover:text-text-primary hover:bg-bg-tertiary rounded-lg transition-colors"
          >
            <ArrowLeft size={20} />
          </button>
          <div>
            <div className="flex items-center gap-2">
              <Target size={20} className="text-primary" />
              <h1 className="text-xl font-bold text-text-primary">{mision.nombre}</h1>
            </div>
            <p className="text-sm text-text-secondary mt-1">{mision.instrucciones}</p>
          </div>
        </div>
      </div>

      {/* Metrics Overview */}
      {metricas && metricas.total_simulaciones > 0 && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <MetricCard
            label="Success Rate"
            value={`${Math.round(metricas.success_rate)}%`}
            color={metricas.success_rate >= 70 ? 'success' : metricas.success_rate >= 40 ? 'warning' : 'error'}
          />
          <MetricCard
            label="Avg Misclicks"
            value={metricas.avg_misclicks.toFixed(1)}
            color={metricas.avg_misclicks <= 1 ? 'success' : metricas.avg_misclicks <= 2 ? 'warning' : 'error'}
          />
          <MetricCard
            label="Avg Pasos"
            value={metricas.avg_pasos.toFixed(1)}
            color="primary"
          />
          <MetricCard
            label="Simulaciones"
            value={metricas.total_simulaciones.toString()}
            color="primary"
          />
        </div>
      )}

      {/* Metrics by Nivel */}
      {metricas && Object.keys(metricas.metrics_by_nivel).length > 0 && (
        <div className="bg-bg-secondary rounded-xl border border-border p-4">
          <h3 className="text-sm font-medium text-text-primary mb-3 flex items-center gap-2">
            <BarChart3 size={16} />
            Metricas por Nivel Digital
          </h3>
          <div className="grid grid-cols-3 gap-4">
            {(['bajo', 'medio', 'alto'] as const).map((nivel) => {
              const data = metricas.metrics_by_nivel[nivel];
              if (!data || data.total === 0) return null;

              return (
                <div key={nivel} className="p-3 bg-bg-tertiary rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <span
                      className={`text-xs font-medium px-2 py-0.5 rounded ${
                        nivel === 'alto'
                          ? 'bg-success/10 text-success'
                          : nivel === 'medio'
                          ? 'bg-warning/10 text-warning'
                          : 'bg-error/10 text-error'
                      }`}
                    >
                      {nivel.toUpperCase()}
                    </span>
                    <span className="text-xs text-text-muted">{data.total} sim.</span>
                  </div>
                  <div className="space-y-1 text-sm">
                    <div className="flex justify-between">
                      <span className="text-text-muted">Success Rate</span>
                      <span className="text-text-primary font-medium">
                        {Math.round(data.success_rate)}%
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-text-muted">Avg Misclicks</span>
                      <span className="text-text-primary font-medium">
                        {data.avg_misclicks.toFixed(1)}
                      </span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Main content */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Simulaciones list */}
        <div className="lg:col-span-1 space-y-4">
          <div className="bg-bg-secondary rounded-xl border border-border overflow-hidden">
            <div className="p-4 border-b border-border">
              <h3 className="font-medium text-text-primary flex items-center gap-2">
                <Users size={16} />
                Simulaciones ({mision.simulaciones.length})
              </h3>
            </div>

            {/* Run new simulation */}
            <div className="p-4 border-b border-border">
              <div className="space-y-2">
                <select
                  value=""
                  onChange={(e) => {
                    if (e.target.value && !selectedArquetipos.includes(e.target.value)) {
                      setSelectedArquetipos([...selectedArquetipos, e.target.value]);
                    }
                  }}
                  className="w-full px-3 py-2 bg-bg-tertiary border border-border rounded-lg text-text-primary text-sm"
                >
                  <option value="">Agregar arquetipo...</option>
                  {arquetipos
                    .filter((a) => !selectedArquetipos.includes(a.id))
                    .map((arq) => (
                      <option key={arq.id} value={arq.id}>
                        {arq.nombre} ({arq.nivel_digital || 'sin nivel'})
                      </option>
                    ))}
                </select>

                {selectedArquetipos.length > 0 && (
                  <div className="flex flex-wrap gap-1">
                    {selectedArquetipos.map((id) => {
                      const arq = arquetipos.find((a) => a.id === id);
                      return (
                        <span
                          key={id}
                          className="inline-flex items-center gap-1 px-2 py-0.5 bg-primary/10 text-primary text-xs rounded"
                        >
                          {arq?.nombre}
                          <button
                            onClick={() =>
                              setSelectedArquetipos(selectedArquetipos.filter((a) => a !== id))
                            }
                            className="hover:text-error"
                          >
                            ×
                          </button>
                        </span>
                      );
                    })}
                  </div>
                )}

                <button
                  onClick={handleRunSimulation}
                  disabled={selectedArquetipos.length === 0 || runningSimulation}
                  className="w-full flex items-center justify-center gap-2 px-3 py-2 bg-primary hover:bg-primary-dark text-white rounded-lg text-sm transition-colors disabled:opacity-50"
                >
                  {runningSimulation ? (
                    <Loader2 size={14} className="animate-spin" />
                  ) : (
                    <Play size={14} />
                  )}
                  Ejecutar simulacion
                </button>
              </div>
            </div>

            {/* Simulaciones list */}
            <div className="max-h-[400px] overflow-y-auto">
              {mision.simulaciones.length === 0 ? (
                <div className="p-8 text-center text-text-muted">
                  <Target size={32} className="mx-auto mb-2 opacity-50" />
                  <p className="text-sm">No hay simulaciones aun</p>
                </div>
              ) : (
                <div className="divide-y divide-border">
                  {mision.simulaciones.map((sim) => (
                    <button
                      key={sim.id}
                      onClick={() => setSelectedSimulacion(sim)}
                      className={`w-full text-left p-4 hover:bg-bg-tertiary transition-colors ${
                        selectedSimulacion?.id === sim.id ? 'bg-bg-tertiary' : ''
                      }`}
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className="font-medium text-text-primary text-sm">
                          {sim.arquetipo?.nombre || 'Arquetipo'}
                        </span>
                        {sim.exito ? (
                          <CheckCircle2 size={16} className="text-success" />
                        ) : (
                          <XCircle size={16} className="text-error" />
                        )}
                      </div>
                      <div className="flex items-center gap-3 text-xs text-text-muted">
                        <span
                          className={`px-1.5 py-0.5 rounded ${
                            sim.arquetipo?.nivel_digital === 'alto'
                              ? 'bg-success/10 text-success'
                              : sim.arquetipo?.nivel_digital === 'medio'
                              ? 'bg-warning/10 text-warning'
                              : 'bg-error/10 text-error'
                          }`}
                        >
                          {sim.arquetipo?.nivel_digital || '-'}
                        </span>
                        <span>{sim.pasos_totales} pasos</span>
                        <span>{sim.misclicks} misclicks</span>
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Simulation Player */}
        <div className="lg:col-span-2">
          {selectedSimulacion ? (
            <div className="space-y-4">
              <SimulationPlayer
                simulacion={selectedSimulacion}
                pantallaImagenes={pantallaImagenes}
              />

              {/* Fricciones */}
              {selectedSimulacion.fricciones.length > 0 && (
                <div className="bg-bg-secondary rounded-xl border border-border p-4">
                  <h4 className="text-sm font-medium text-text-primary mb-3 flex items-center gap-2">
                    <AlertTriangle size={16} className="text-warning" />
                    Fricciones detectadas ({selectedSimulacion.fricciones.length})
                  </h4>
                  <div className="space-y-2">
                    {selectedSimulacion.fricciones.map((friccion, idx) => (
                      <div
                        key={idx}
                        className="p-3 bg-warning/5 border border-warning/20 rounded-lg"
                      >
                        <span className="text-xs font-medium text-warning uppercase">
                          {friccion.tipo}
                        </span>
                        <p className="text-sm text-text-secondary mt-1">
                          {friccion.descripcion}
                        </p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Feedback */}
              {selectedSimulacion.feedback_arquetipo && (
                <div className="bg-bg-secondary rounded-xl border border-border p-4">
                  <h4 className="text-sm font-medium text-text-primary mb-2">
                    Feedback del Usuario Sintetico
                  </h4>
                  <p className="text-text-secondary italic">
                    "{selectedSimulacion.feedback_arquetipo}"
                  </p>
                </div>
              )}
            </div>
          ) : (
            <div className="h-full flex items-center justify-center bg-bg-secondary rounded-xl border border-border">
              <div className="text-center text-text-muted py-12">
                <Play size={48} className="mx-auto mb-3 opacity-50" />
                <p>Selecciona una simulacion para ver el recorrido</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

interface MetricCardProps {
  label: string;
  value: string;
  color: 'success' | 'warning' | 'error' | 'primary';
}

function MetricCard({ label, value, color }: MetricCardProps) {
  const colorClasses = {
    success: 'bg-success/10 text-success border-success/20',
    warning: 'bg-warning/10 text-warning border-warning/20',
    error: 'bg-error/10 text-error border-error/20',
    primary: 'bg-primary/10 text-primary border-primary/20',
  };

  return (
    <div className={`p-4 rounded-xl border ${colorClasses[color]}`}>
      <p className="text-xs uppercase tracking-wide opacity-75">{label}</p>
      <p className="text-2xl font-bold mt-1">{value}</p>
    </div>
  );
}
