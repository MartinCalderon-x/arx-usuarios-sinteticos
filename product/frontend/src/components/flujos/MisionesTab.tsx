import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Target,
  Plus,
  Play,
  Trash2,
  Edit2,
  Loader2,
  CheckCircle2,
  Clock,
  Users,
} from 'lucide-react';
import { usabilityApi, arquetiposApi, type Mision, type MisionCreate, type Pantalla, type Arquetipo } from '../../lib/api';
import { MisionForm } from './MisionForm';

interface MisionesTabProps {
  flujoId: string;
  pantallas: Pantalla[];
}

export function MisionesTab({ flujoId, pantallas }: MisionesTabProps) {
  const navigate = useNavigate();
  const [misiones, setMisiones] = useState<Mision[]>([]);
  const [arquetipos, setArquetipos] = useState<Arquetipo[]>([]);
  const [loading, setLoading] = useState(true);
  const [showForm, setShowForm] = useState(false);
  const [editingMision, setEditingMision] = useState<Mision | null>(null);
  const [runningMision, setRunningMision] = useState<string | null>(null);
  const [selectedArquetipos, setSelectedArquetipos] = useState<string[]>([]);
  const [showArquetipoModal, setShowArquetipoModal] = useState<string | null>(null);

  useEffect(() => {
    loadData();
  }, [flujoId]);

  async function loadData() {
    setLoading(true);
    try {
      const [misionesRes, arquetiposRes] = await Promise.all([
        usabilityApi.listMisiones(flujoId),
        arquetiposApi.list(),
      ]);
      setMisiones(misionesRes.misiones);
      setArquetipos(arquetiposRes.arquetipos);
    } catch (error) {
      console.error('Error loading data:', error);
    } finally {
      setLoading(false);
    }
  }

  async function handleCreateMision(data: MisionCreate) {
    await usabilityApi.createMision(flujoId, data);
    await loadData();
  }

  async function handleUpdateMision(data: MisionCreate) {
    if (!editingMision) return;
    await usabilityApi.updateMision(flujoId, editingMision.id, data);
    setEditingMision(null);
    await loadData();
  }

  async function handleDeleteMision(misionId: string) {
    if (!confirm('Estas seguro de eliminar esta mision y todas sus simulaciones?')) return;
    try {
      await usabilityApi.deleteMision(flujoId, misionId);
      await loadData();
    } catch (error) {
      console.error('Error deleting mision:', error);
    }
  }

  async function handleRunSimulation(misionId: string) {
    if (selectedArquetipos.length === 0) {
      alert('Selecciona al menos un arquetipo');
      return;
    }

    setRunningMision(misionId);
    setShowArquetipoModal(null);

    try {
      await usabilityApi.runSimulation(misionId, selectedArquetipos);
      setSelectedArquetipos([]);
      // Navigate to mision detail to see results
      navigate(`/flujos/${flujoId}/misiones/${misionId}`);
    } catch (error) {
      console.error('Error running simulation:', error);
      alert('Error al ejecutar simulacion');
    } finally {
      setRunningMision(null);
    }
  }

  function toggleArquetipo(arquetipoId: string) {
    setSelectedArquetipos((prev) =>
      prev.includes(arquetipoId)
        ? prev.filter((id) => id !== arquetipoId)
        : [...prev, arquetipoId]
    );
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 size={24} className="animate-spin text-primary" />
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold text-text-primary">Misiones de Usability</h3>
          <p className="text-sm text-text-secondary">
            Define tareas y simula usuarios navegando tu flujo
          </p>
        </div>
        <button
          onClick={() => setShowForm(true)}
          className="flex items-center gap-2 px-4 py-2 bg-primary hover:bg-primary-dark text-white rounded-lg transition-colors"
        >
          <Plus size={18} />
          Nueva mision
        </button>
      </div>

      {/* Misiones list */}
      {misiones.length === 0 ? (
        <div className="text-center py-12 bg-bg-secondary rounded-xl border border-border">
          <Target size={48} className="mx-auto text-text-muted mb-3" />
          <h4 className="text-lg font-medium text-text-primary mb-1">Sin misiones</h4>
          <p className="text-text-secondary mb-4">
            Crea una mision para simular usuarios navegando tu flujo
          </p>
          <button
            onClick={() => setShowForm(true)}
            className="inline-flex items-center gap-2 px-4 py-2 bg-primary hover:bg-primary-dark text-white rounded-lg transition-colors"
          >
            <Plus size={18} />
            Crear primera mision
          </button>
        </div>
      ) : (
        <div className="space-y-3">
          {misiones.map((mision) => (
            <div
              key={mision.id}
              className="p-4 bg-bg-secondary rounded-xl border border-border hover:border-primary/30 transition-colors"
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <Target size={16} className="text-primary" />
                    <h4 className="font-medium text-text-primary">{mision.nombre}</h4>
                    <span
                      className={`text-xs px-2 py-0.5 rounded ${
                        mision.estado === 'activa'
                          ? 'bg-success/10 text-success'
                          : 'bg-text-muted/10 text-text-muted'
                      }`}
                    >
                      {mision.estado}
                    </span>
                  </div>
                  <p className="text-sm text-text-secondary mb-2">{mision.instrucciones}</p>

                  <div className="flex items-center gap-4 text-xs text-text-muted">
                    <span className="flex items-center gap-1">
                      <Clock size={12} />
                      Max {mision.max_pasos} pasos
                    </span>
                    {mision.elemento_objetivo?.texto && (
                      <span>Objetivo: {mision.elemento_objetivo.texto}</span>
                    )}
                  </div>
                </div>

                <div className="flex items-center gap-2 ml-4">
                  <button
                    onClick={() => navigate(`/flujos/${flujoId}/misiones/${mision.id}`)}
                    className="p-2 text-text-muted hover:text-text-primary hover:bg-bg-tertiary rounded-lg transition-colors"
                    title="Ver detalles"
                  >
                    <CheckCircle2 size={18} />
                  </button>
                  <button
                    onClick={() => setShowArquetipoModal(mision.id)}
                    disabled={runningMision === mision.id}
                    className="flex items-center gap-1.5 px-3 py-1.5 bg-primary hover:bg-primary-dark text-white rounded-lg transition-colors disabled:opacity-50"
                    title="Ejecutar simulacion"
                  >
                    {runningMision === mision.id ? (
                      <Loader2 size={14} className="animate-spin" />
                    ) : (
                      <Play size={14} />
                    )}
                    Simular
                  </button>
                  <button
                    onClick={() => {
                      setEditingMision(mision);
                      setShowForm(true);
                    }}
                    className="p-2 text-text-muted hover:text-text-primary hover:bg-bg-tertiary rounded-lg transition-colors"
                    title="Editar"
                  >
                    <Edit2 size={16} />
                  </button>
                  <button
                    onClick={() => handleDeleteMision(mision.id)}
                    className="p-2 text-text-muted hover:text-error hover:bg-error/10 rounded-lg transition-colors"
                    title="Eliminar"
                  >
                    <Trash2 size={16} />
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Mision Form Modal */}
      <MisionForm
        isOpen={showForm}
        onClose={() => {
          setShowForm(false);
          setEditingMision(null);
        }}
        onSubmit={editingMision ? handleUpdateMision : handleCreateMision}
        pantallas={pantallas}
        mision={editingMision}
      />

      {/* Arquetipo Selection Modal */}
      {showArquetipoModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-bg-secondary rounded-xl border border-border w-full max-w-md mx-4 overflow-hidden">
            <div className="p-4 border-b border-border">
              <div className="flex items-center gap-2">
                <Users size={20} className="text-primary" />
                <h3 className="text-lg font-semibold text-text-primary">
                  Seleccionar Arquetipos
                </h3>
              </div>
              <p className="text-sm text-text-secondary mt-1">
                Elige los usuarios sinteticos para la simulacion
              </p>
            </div>

            <div className="p-4 max-h-[400px] overflow-y-auto">
              {arquetipos.length === 0 ? (
                <div className="text-center py-8 text-text-muted">
                  <p>No hay arquetipos disponibles</p>
                  <button
                    onClick={() => navigate('/arquetipos/nuevo')}
                    className="text-primary hover:underline mt-2 text-sm"
                  >
                    Crear arquetipo
                  </button>
                </div>
              ) : (
                <div className="space-y-2">
                  {arquetipos.map((arq) => (
                    <button
                      key={arq.id}
                      onClick={() => toggleArquetipo(arq.id)}
                      className={`w-full text-left p-3 rounded-lg border transition-colors ${
                        selectedArquetipos.includes(arq.id)
                          ? 'border-primary bg-primary/10'
                          : 'border-border hover:bg-bg-tertiary'
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <div>
                          <span className="font-medium text-text-primary">{arq.nombre}</span>
                          {arq.nivel_digital && (
                            <span
                              className={`ml-2 text-xs px-2 py-0.5 rounded ${
                                arq.nivel_digital === 'alto'
                                  ? 'bg-success/10 text-success'
                                  : arq.nivel_digital === 'medio'
                                  ? 'bg-warning/10 text-warning'
                                  : 'bg-error/10 text-error'
                              }`}
                            >
                              {arq.nivel_digital}
                            </span>
                          )}
                        </div>
                        <div
                          className={`w-5 h-5 rounded border-2 flex items-center justify-center transition-colors ${
                            selectedArquetipos.includes(arq.id)
                              ? 'border-primary bg-primary'
                              : 'border-text-muted'
                          }`}
                        >
                          {selectedArquetipos.includes(arq.id) && (
                            <CheckCircle2 size={14} className="text-white" />
                          )}
                        </div>
                      </div>
                      {arq.ocupacion && (
                        <p className="text-xs text-text-muted mt-1">{arq.ocupacion}</p>
                      )}
                    </button>
                  ))}
                </div>
              )}
            </div>

            <div className="p-4 border-t border-border flex justify-between items-center">
              <span className="text-sm text-text-muted">
                {selectedArquetipos.length} seleccionado(s)
              </span>
              <div className="flex gap-2">
                <button
                  onClick={() => {
                    setShowArquetipoModal(null);
                    setSelectedArquetipos([]);
                  }}
                  className="px-4 py-2 text-text-secondary hover:text-text-primary hover:bg-bg-tertiary rounded-lg transition-colors"
                >
                  Cancelar
                </button>
                <button
                  onClick={() => handleRunSimulation(showArquetipoModal)}
                  disabled={selectedArquetipos.length === 0}
                  className="flex items-center gap-2 px-4 py-2 bg-primary hover:bg-primary-dark text-white rounded-lg transition-colors disabled:opacity-50"
                >
                  <Play size={16} />
                  Ejecutar ({selectedArquetipos.length})
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
