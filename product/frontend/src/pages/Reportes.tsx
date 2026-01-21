import { useEffect, useState } from 'react';
import { FileText, Download, Trash2, Plus, Loader2 } from 'lucide-react';
import { reportesApi, arquetiposApi, analisisApi, type Reporte, type Arquetipo, type Analisis } from '../lib/api';

export function Reportes() {
  const [reportes, setReportes] = useState<Reporte[]>([]);
  const [loading, setLoading] = useState(true);
  const [showModal, setShowModal] = useState(false);

  useEffect(() => {
    loadReportes();
  }, []);

  async function loadReportes() {
    try {
      const { reportes } = await reportesApi.list();
      setReportes(reportes);
    } catch (error) {
      console.error('Error loading reportes:', error);
    } finally {
      setLoading(false);
    }
  }

  async function handleDownload(id: string, titulo: string, formato: string) {
    try {
      const blob = await reportesApi.download(id);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${titulo}.${formato}`;
      a.click();
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Error downloading:', error);
      alert('Error al descargar el reporte');
    }
  }

  async function handleDelete(id: string) {
    if (!confirm('Estas seguro de eliminar este reporte?')) return;
    try {
      await reportesApi.delete(id);
      setReportes(reportes.filter(r => r.id !== id));
    } catch (error) {
      console.error('Error deleting:', error);
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-text-primary">Reportes</h1>
          <p className="text-text-secondary mt-1">Genera y descarga reportes PDF/PPT</p>
        </div>
        <button
          onClick={() => setShowModal(true)}
          className="flex items-center gap-2 px-4 py-2 bg-success hover:bg-success/90 text-white rounded-lg transition-colors"
        >
          <Plus size={18} />
          Generar reporte
        </button>
      </div>

      {/* List */}
      {loading ? (
        <div className="space-y-3">
          {[1, 2, 3].map(i => (
            <div key={i} className="bg-bg-secondary rounded-xl p-4 animate-pulse flex items-center gap-4">
              <div className="w-12 h-12 bg-bg-tertiary rounded-lg" />
              <div className="flex-1">
                <div className="h-5 bg-bg-tertiary rounded w-1/3 mb-2" />
                <div className="h-4 bg-bg-tertiary rounded w-1/4" />
              </div>
            </div>
          ))}
        </div>
      ) : reportes.length === 0 ? (
        <div className="text-center py-12 bg-bg-secondary rounded-xl border border-border">
          <FileText size={48} className="mx-auto text-text-muted mb-4" />
          <h3 className="text-lg font-medium text-text-primary mb-2">No hay reportes</h3>
          <p className="text-text-secondary mb-4">Genera tu primer reporte</p>
          <button
            onClick={() => setShowModal(true)}
            className="inline-flex items-center gap-2 px-4 py-2 bg-success hover:bg-success/90 text-white rounded-lg transition-colors"
          >
            <Plus size={18} />
            Generar reporte
          </button>
        </div>
      ) : (
        <div className="space-y-3">
          {reportes.map(reporte => (
            <div
              key={reporte.id}
              className="bg-bg-secondary rounded-xl p-4 border border-border hover:border-border-dark transition-colors flex items-center gap-4"
            >
              <div className={`p-3 rounded-lg ${reporte.formato === 'pdf' ? 'bg-error/10' : 'bg-warning/10'}`}>
                <FileText size={24} className={reporte.formato === 'pdf' ? 'text-error' : 'text-warning'} />
              </div>
              <div className="flex-1">
                <h3 className="font-medium text-text-primary">{reporte.titulo}</h3>
                <p className="text-sm text-text-muted">
                  {reporte.formato.toUpperCase()} - {reporte.created_at ? new Date(reporte.created_at).toLocaleDateString() : 'Sin fecha'}
                </p>
              </div>
              <div className="flex gap-2">
                <button
                  onClick={() => handleDownload(reporte.id, reporte.titulo, reporte.formato)}
                  className="p-2 text-text-muted hover:text-success hover:bg-success/10 rounded-lg transition-colors"
                >
                  <Download size={18} />
                </button>
                <button
                  onClick={() => handleDelete(reporte.id)}
                  className="p-2 text-text-muted hover:text-error hover:bg-error/10 rounded-lg transition-colors"
                >
                  <Trash2 size={18} />
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Modal */}
      {showModal && (
        <ReporteModal
          onClose={() => setShowModal(false)}
          onCreated={() => {
            setShowModal(false);
            loadReportes();
          }}
        />
      )}
    </div>
  );
}

function ReporteModal({ onClose, onCreated }: { onClose: () => void; onCreated: () => void }) {
  const [titulo, setTitulo] = useState('');
  const [formato, setFormato] = useState<'pdf' | 'pptx'>('pdf');
  const [arquetipos, setArquetipos] = useState<Arquetipo[]>([]);
  const [analisis, setAnalisis] = useState<Analisis[]>([]);
  const [selectedArquetipos, setSelectedArquetipos] = useState<string[]>([]);
  const [selectedAnalisis, setSelectedAnalisis] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [loadingData, setLoadingData] = useState(true);

  useEffect(() => {
    async function loadData() {
      try {
        const [arq, ana] = await Promise.all([
          arquetiposApi.list(),
          analisisApi.list(),
        ]);
        setArquetipos(arq.arquetipos);
        setAnalisis(ana.analisis);
      } catch (error) {
        console.error('Error loading data:', error);
      } finally {
        setLoadingData(false);
      }
    }
    loadData();
  }, []);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!titulo) return;

    setLoading(true);
    try {
      await reportesApi.generate({
        titulo,
        formato,
        arquetipos_ids: selectedArquetipos.length > 0 ? selectedArquetipos : undefined,
        analisis_ids: selectedAnalisis.length > 0 ? selectedAnalisis : undefined,
      });
      onCreated();
    } catch (error) {
      console.error('Error generating report:', error);
      alert('Error al generar el reporte');
    } finally {
      setLoading(false);
    }
  }

  function toggleSelection(id: string, list: string[], setList: (v: string[]) => void) {
    if (list.includes(id)) {
      setList(list.filter(i => i !== id));
    } else {
      setList([...list, id]);
    }
  }

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-bg-primary rounded-xl max-w-lg w-full max-h-[80vh] overflow-auto">
        <div className="p-6 border-b border-border">
          <h2 className="text-xl font-semibold text-text-primary">Generar reporte</h2>
        </div>

        <form onSubmit={handleSubmit} className="p-6 space-y-4">
          <div>
            <label className="block text-sm font-medium text-text-secondary mb-1.5">Titulo</label>
            <input
              type="text"
              value={titulo}
              onChange={(e) => setTitulo(e.target.value)}
              className="w-full px-4 py-2.5 rounded-lg border border-border bg-bg-secondary text-text-primary focus:outline-none focus:ring-2 focus:ring-primary/50"
              placeholder="Mi reporte"
              required
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-text-secondary mb-1.5">Formato</label>
            <div className="flex gap-2">
              <button
                type="button"
                onClick={() => setFormato('pdf')}
                className={`flex-1 px-4 py-2 rounded-lg border transition-colors ${
                  formato === 'pdf' ? 'border-primary bg-primary/10 text-primary' : 'border-border text-text-secondary'
                }`}
              >
                PDF
              </button>
              <button
                type="button"
                onClick={() => setFormato('pptx')}
                className={`flex-1 px-4 py-2 rounded-lg border transition-colors ${
                  formato === 'pptx' ? 'border-primary bg-primary/10 text-primary' : 'border-border text-text-secondary'
                }`}
              >
                PowerPoint
              </button>
            </div>
          </div>

          {loadingData ? (
            <div className="py-4 text-center text-text-muted">Cargando datos...</div>
          ) : (
            <>
              {arquetipos.length > 0 && (
                <div>
                  <label className="block text-sm font-medium text-text-secondary mb-1.5">
                    Arquetipos ({selectedArquetipos.length} seleccionados)
                  </label>
                  <div className="flex flex-wrap gap-2">
                    {arquetipos.map(a => (
                      <button
                        key={a.id}
                        type="button"
                        onClick={() => toggleSelection(a.id, selectedArquetipos, setSelectedArquetipos)}
                        className={`px-3 py-1.5 text-sm rounded-lg border transition-colors ${
                          selectedArquetipos.includes(a.id)
                            ? 'border-primary bg-primary/10 text-primary'
                            : 'border-border text-text-secondary hover:border-border-dark'
                        }`}
                      >
                        {a.nombre}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {analisis.length > 0 && (
                <div>
                  <label className="block text-sm font-medium text-text-secondary mb-1.5">
                    Analisis ({selectedAnalisis.length} seleccionados)
                  </label>
                  <div className="flex flex-wrap gap-2">
                    {analisis.slice(0, 10).map(a => (
                      <button
                        key={a.id}
                        type="button"
                        onClick={() => toggleSelection(a.id, selectedAnalisis, setSelectedAnalisis)}
                        className={`px-3 py-1.5 text-sm rounded-lg border transition-colors ${
                          selectedAnalisis.includes(a.id)
                            ? 'border-secondary bg-secondary/10 text-secondary'
                            : 'border-border text-text-secondary hover:border-border-dark'
                        }`}
                      >
                        {a.id.slice(0, 8)}...
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}

          <div className="flex justify-end gap-3 pt-4 border-t border-border">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-text-secondary hover:text-text-primary hover:bg-bg-tertiary rounded-lg transition-colors"
            >
              Cancelar
            </button>
            <button
              type="submit"
              disabled={loading || !titulo}
              className="flex items-center gap-2 px-4 py-2 bg-success hover:bg-success/90 text-white rounded-lg transition-colors disabled:opacity-50"
            >
              {loading ? <Loader2 size={18} className="animate-spin" /> : <FileText size={18} />}
              Generar
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
