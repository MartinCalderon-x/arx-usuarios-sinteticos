import { useEffect, useState, useCallback } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import {
  ArrowLeft,
  Plus,
  Trash2,
  Loader2,
  Eye,
  Flame,
  Layers,
  Image as ImageIcon,
  Link as LinkIcon,
  ChevronLeft,
  ChevronRight,
  X,
  Upload,
  Target,
  BarChart3,
  Scan,
  MousePointer2,
} from 'lucide-react';
import { flujosApi, usabilityApi, type FlujoDetail as FlujoDetailType, type Pantalla } from '../lib/api';
import { MisionesTab } from '../components/flujos/MisionesTab';
import { ElementOverlay } from '../components/flujos/ElementOverlay';

type ViewMode = 'screenshot' | 'heatmap' | 'overlay' | 'elements';
type TabMode = 'pantallas' | 'misiones';

export function FlujoDetail() {
  const { id } = useParams();
  const navigate = useNavigate();

  const [flujo, setFlujo] = useState<FlujoDetailType | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedPantalla, setSelectedPantalla] = useState<Pantalla | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>('overlay');
  const [showAddModal, setShowAddModal] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [activeTab, setActiveTab] = useState<TabMode>('pantallas');
  const [detectingElements, setDetectingElements] = useState(false);

  useEffect(() => {
    if (id) loadFlujo(id);
  }, [id]);

  async function loadFlujo(flujoId: string) {
    setLoading(true);
    try {
      const data = await flujosApi.get(flujoId);
      setFlujo(data);
      // Select first pantalla by default
      if (data.pantallas.length > 0 && !selectedPantalla) {
        setSelectedPantalla(data.pantallas[0]);
      }
    } catch (error) {
      console.error('Error loading flujo:', error);
      navigate('/flujos');
    } finally {
      setLoading(false);
    }
  }

  async function handleDeletePantalla(pantallaId: string) {
    if (!id || !confirm('Estas seguro de eliminar esta pantalla?')) return;
    try {
      await flujosApi.deletePantalla(id, pantallaId);
      // Reload flujo to get updated list
      await loadFlujo(id);
      // Clear selection if deleted pantalla was selected
      if (selectedPantalla?.id === pantallaId) {
        setSelectedPantalla(flujo?.pantallas[0] || null);
      }
    } catch (error) {
      console.error('Error deleting pantalla:', error);
    }
  }

  async function handleFileUpload(files: FileList | null) {
    if (!files || files.length === 0 || !id) return;

    setUploading(true);
    try {
      for (const file of Array.from(files)) {
        await flujosApi.uploadPantalla(id, file);
      }
      await loadFlujo(id);
      setShowAddModal(false);
    } catch (error) {
      console.error('Error uploading pantalla:', error);
      alert('Error al subir la imagen');
    } finally {
      setUploading(false);
    }
  }

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    handleFileUpload(e.dataTransfer.files);
  }, [id]);

  async function handleDetectElements() {
    if (!selectedPantalla || !id) return;
    setDetectingElements(true);
    try {
      const result = await usabilityApi.detectarElementos(selectedPantalla.id);
      // Update the pantalla in local state
      if (flujo) {
        const updatedPantallas = flujo.pantallas.map((p) =>
          p.id === selectedPantalla.id
            ? { ...p, elementos_clickeables: result.elementos }
            : p
        );
        setFlujo({ ...flujo, pantallas: updatedPantallas });
        setSelectedPantalla({
          ...selectedPantalla,
          elementos_clickeables: result.elementos,
        });
      }
    } catch (error) {
      console.error('Error detecting elements:', error);
      alert('Error al detectar elementos');
    } finally {
      setDetectingElements(false);
    }
  }

  function getCurrentImageUrl(): string | undefined {
    if (!selectedPantalla) return undefined;
    switch (viewMode) {
      case 'screenshot':
        return selectedPantalla.screenshot_url;
      case 'heatmap':
        return selectedPantalla.heatmap_url;
      case 'overlay':
        return selectedPantalla.overlay_url || selectedPantalla.heatmap_url;
      default:
        return selectedPantalla.screenshot_url;
    }
  }

  function navigatePantalla(direction: 'prev' | 'next') {
    if (!flujo || !selectedPantalla) return;
    const currentIndex = flujo.pantallas.findIndex(p => p.id === selectedPantalla.id);
    const newIndex = direction === 'prev' ? currentIndex - 1 : currentIndex + 1;
    if (newIndex >= 0 && newIndex < flujo.pantallas.length) {
      setSelectedPantalla(flujo.pantallas[newIndex]);
    }
  }

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

  const currentIndex = selectedPantalla
    ? flujo.pantallas.findIndex(p => p.id === selectedPantalla.id)
    : -1;

  return (
    <div className="h-[calc(100vh-3rem)] flex flex-col">
      {/* Header */}
      <div className="pb-4 border-b border-border flex-shrink-0">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-4">
            <button
              onClick={() => navigate('/flujos')}
              className="p-2 text-text-secondary hover:text-text-primary hover:bg-bg-tertiary rounded-lg transition-colors"
            >
              <ArrowLeft size={20} />
            </button>
            <div>
              <h1 className="text-xl font-bold text-text-primary">{flujo.nombre}</h1>
              <p className="text-sm text-text-secondary">
                {flujo.total_pantallas} {flujo.total_pantallas === 1 ? 'pantalla' : 'pantallas'}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => navigate(`/flujos/${id}/usability`)}
              className="flex items-center gap-2 px-3 py-2 text-text-secondary hover:text-text-primary hover:bg-bg-tertiary rounded-lg transition-colors"
            >
              <BarChart3 size={18} />
              Dashboard
            </button>
            {activeTab === 'pantallas' && (
              <button
                onClick={() => setShowAddModal(true)}
                className="flex items-center gap-2 px-4 py-2 bg-primary hover:bg-primary-dark text-white rounded-lg transition-colors"
              >
                <Plus size={18} />
                Agregar pantalla
              </button>
            )}
          </div>
        </div>

        {/* Tabs */}
        <div className="flex gap-1">
          <button
            onClick={() => setActiveTab('pantallas')}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              activeTab === 'pantallas'
                ? 'bg-primary text-white'
                : 'text-text-secondary hover:bg-bg-tertiary'
            }`}
          >
            <Layers size={16} />
            Pantallas
          </button>
          <button
            onClick={() => setActiveTab('misiones')}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              activeTab === 'misiones'
                ? 'bg-primary text-white'
                : 'text-text-secondary hover:bg-bg-tertiary'
            }`}
          >
            <Target size={16} />
            Misiones
          </button>
        </div>
      </div>

      {/* Main content */}
      {activeTab === 'misiones' ? (
        <div className="flex-1 pt-4 min-h-0 overflow-y-auto">
          <MisionesTab flujoId={id!} pantallas={flujo.pantallas} />
        </div>
      ) : (
      /* Pantallas - 3 column layout */
      <div className="flex-1 flex gap-4 pt-4 min-h-0">
        {/* Column 1: Timeline */}
        <div className="w-48 flex-shrink-0 bg-bg-secondary rounded-xl border border-border overflow-hidden flex flex-col">
          <div className="p-3 border-b border-border">
            <h3 className="text-sm font-medium text-text-primary">Pantallas</h3>
          </div>
          <div className="flex-1 overflow-y-auto p-2 space-y-2">
            {flujo.pantallas.length === 0 ? (
              <div className="text-center py-8 px-2">
                <Layers size={24} className="mx-auto text-text-muted mb-2" />
                <p className="text-xs text-text-muted">Sin pantallas</p>
              </div>
            ) : (
              flujo.pantallas.map((pantalla, index) => (
                <button
                  key={pantalla.id}
                  onClick={() => setSelectedPantalla(pantalla)}
                  className={`w-full text-left p-2 rounded-lg transition-colors group relative ${
                    selectedPantalla?.id === pantalla.id
                      ? 'bg-primary/10 border border-primary'
                      : 'hover:bg-bg-tertiary border border-transparent'
                  }`}
                >
                  <div className="flex items-center gap-2 mb-1">
                    <span className="w-5 h-5 flex items-center justify-center bg-bg-tertiary rounded text-xs font-medium text-text-secondary">
                      {index + 1}
                    </span>
                    <span className="text-xs text-text-muted truncate flex-1">
                      {pantalla.origen === 'upload' ? (
                        <ImageIcon size={12} className="inline mr-1" />
                      ) : (
                        <LinkIcon size={12} className="inline mr-1" />
                      )}
                      {pantalla.origen}
                    </span>
                  </div>
                  {pantalla.screenshot_url && (
                    <div className="aspect-video bg-bg-tertiary rounded overflow-hidden">
                      <img
                        src={pantalla.screenshot_url}
                        alt={pantalla.titulo || `Pantalla ${index + 1}`}
                        className="w-full h-full object-cover"
                      />
                    </div>
                  )}
                  <p className="text-xs text-text-secondary mt-1 truncate">
                    {pantalla.titulo || `Pantalla ${index + 1}`}
                  </p>
                  {/* Delete button on hover */}
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDeletePantalla(pantalla.id);
                    }}
                    className="absolute top-2 right-2 p-1 text-text-muted hover:text-error hover:bg-error/10 rounded opacity-0 group-hover:opacity-100 transition-opacity"
                    title="Eliminar"
                  >
                    <Trash2 size={12} />
                  </button>
                </button>
              ))
            )}
          </div>
        </div>

        {/* Column 2: Visor */}
        <div className="flex-1 bg-bg-secondary rounded-xl border border-border overflow-hidden flex flex-col min-w-0">
          {selectedPantalla ? (
            <>
              {/* View mode tabs */}
              <div className="flex items-center justify-between p-3 border-b border-border">
                <div className="flex gap-1">
                  <button
                    onClick={() => setViewMode('screenshot')}
                    className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm transition-colors ${
                      viewMode === 'screenshot'
                        ? 'bg-primary text-white'
                        : 'text-text-secondary hover:bg-bg-tertiary'
                    }`}
                  >
                    <ImageIcon size={14} />
                    Original
                  </button>
                  <button
                    onClick={() => setViewMode('heatmap')}
                    className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm transition-colors ${
                      viewMode === 'heatmap'
                        ? 'bg-primary text-white'
                        : 'text-text-secondary hover:bg-bg-tertiary'
                    }`}
                  >
                    <Flame size={14} />
                    Heatmap
                  </button>
                  <button
                    onClick={() => setViewMode('overlay')}
                    className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm transition-colors ${
                      viewMode === 'overlay'
                        ? 'bg-primary text-white'
                        : 'text-text-secondary hover:bg-bg-tertiary'
                    }`}
                  >
                    <Eye size={14} />
                    Overlay
                  </button>
                  <button
                    onClick={() => setViewMode('elements')}
                    className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm transition-colors ${
                      viewMode === 'elements'
                        ? 'bg-primary text-white'
                        : 'text-text-secondary hover:bg-bg-tertiary'
                    }`}
                  >
                    <MousePointer2 size={14} />
                    Elementos
                  </button>
                  {/* Detect elements button */}
                  <button
                    onClick={handleDetectElements}
                    disabled={detectingElements}
                    className="flex items-center gap-1.5 px-3 py-1.5 ml-2 bg-bg-tertiary hover:bg-primary/10 text-text-secondary hover:text-primary rounded-lg text-sm transition-colors disabled:opacity-50"
                    title="Detectar elementos clickeables"
                  >
                    {detectingElements ? (
                      <Loader2 size={14} className="animate-spin" />
                    ) : (
                      <Scan size={14} />
                    )}
                    Detectar
                  </button>
                </div>
                {/* Navigation */}
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => navigatePantalla('prev')}
                    disabled={currentIndex <= 0}
                    className="p-1.5 text-text-muted hover:text-text-primary hover:bg-bg-tertiary rounded transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
                  >
                    <ChevronLeft size={18} />
                  </button>
                  <span className="text-sm text-text-secondary">
                    {currentIndex + 1} / {flujo.pantallas.length}
                  </span>
                  <button
                    onClick={() => navigatePantalla('next')}
                    disabled={currentIndex >= flujo.pantallas.length - 1}
                    className="p-1.5 text-text-muted hover:text-text-primary hover:bg-bg-tertiary rounded transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
                  >
                    <ChevronRight size={18} />
                  </button>
                </div>
              </div>
              {/* Image viewer */}
              <div className="flex-1 p-4 flex items-center justify-center bg-bg-tertiary/30 overflow-auto">
                {viewMode === 'elements' && selectedPantalla.screenshot_url ? (
                  <ElementOverlay
                    imageUrl={selectedPantalla.screenshot_url}
                    elementos={selectedPantalla.elementos_clickeables || []}
                    showLabels={true}
                  />
                ) : getCurrentImageUrl() ? (
                  <img
                    src={getCurrentImageUrl()}
                    alt={selectedPantalla.titulo || 'Pantalla'}
                    className="max-w-full max-h-full object-contain rounded-lg shadow-lg"
                  />
                ) : (
                  <div className="text-center text-text-muted">
                    <ImageIcon size={48} className="mx-auto mb-2 opacity-50" />
                    <p>Imagen no disponible</p>
                  </div>
                )}
              </div>
            </>
          ) : (
            <div className="flex-1 flex items-center justify-center">
              <div className="text-center text-text-muted">
                <Eye size={48} className="mx-auto mb-2 opacity-50" />
                <p>Selecciona una pantalla para ver el analisis</p>
              </div>
            </div>
          )}
        </div>

        {/* Column 3: Analysis */}
        <div className="w-72 flex-shrink-0 bg-bg-secondary rounded-xl border border-border overflow-hidden flex flex-col">
          <div className="p-3 border-b border-border">
            <h3 className="text-sm font-medium text-text-primary">Analisis</h3>
          </div>
          {selectedPantalla ? (
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              {/* Clarity Score */}
              {selectedPantalla.clarity_score !== undefined && (
                <div>
                  <h4 className="text-xs font-medium text-text-muted uppercase tracking-wide mb-2">
                    Clarity Score
                  </h4>
                  <div className="flex items-center gap-3">
                    <div className="flex-1 h-2 bg-bg-tertiary rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all ${
                          selectedPantalla.clarity_score >= 70
                            ? 'bg-success'
                            : selectedPantalla.clarity_score >= 40
                            ? 'bg-warning'
                            : 'bg-error'
                        }`}
                        style={{ width: `${selectedPantalla.clarity_score}%` }}
                      />
                    </div>
                    <span className="text-lg font-bold text-text-primary">
                      {Math.round(selectedPantalla.clarity_score)}
                    </span>
                  </div>
                </div>
              )}

              {/* Areas of Interest */}
              {selectedPantalla.areas_interes && selectedPantalla.areas_interes.length > 0 && (
                <div>
                  <h4 className="text-xs font-medium text-text-muted uppercase tracking-wide mb-2">
                    Areas de Interes
                  </h4>
                  <div className="space-y-2">
                    {selectedPantalla.areas_interes.map((area, idx) => (
                      <div
                        key={idx}
                        className="flex items-center justify-between p-2 bg-bg-tertiary rounded-lg"
                      >
                        <div className="flex items-center gap-2">
                          <span className="w-5 h-5 flex items-center justify-center bg-primary/10 text-primary rounded text-xs font-medium">
                            {area.orden_visual}
                          </span>
                          <span className="text-sm text-text-primary">{area.nombre}</span>
                        </div>
                        <span className="text-xs text-text-muted">{area.intensidad}%</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Insights */}
              {selectedPantalla.insights && selectedPantalla.insights.length > 0 && (
                <div>
                  <h4 className="text-xs font-medium text-text-muted uppercase tracking-wide mb-2">
                    Insights
                  </h4>
                  <ul className="space-y-2">
                    {selectedPantalla.insights.map((insight, idx) => (
                      <li
                        key={idx}
                        className="text-sm text-text-secondary pl-3 border-l-2 border-primary/30"
                      >
                        {insight}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Model Used */}
              {selectedPantalla.modelo_usado && (
                <div>
                  <h4 className="text-xs font-medium text-text-muted uppercase tracking-wide mb-2">
                    Modelo
                  </h4>
                  <span className="text-xs px-2 py-1 bg-bg-tertiary text-text-secondary rounded">
                    {selectedPantalla.modelo_usado}
                  </span>
                </div>
              )}

              {/* No analysis data */}
              {!selectedPantalla.clarity_score &&
                (!selectedPantalla.areas_interes || selectedPantalla.areas_interes.length === 0) &&
                (!selectedPantalla.insights || selectedPantalla.insights.length === 0) && (
                  <div className="text-center py-8 text-text-muted">
                    <p className="text-sm">No hay datos de analisis disponibles</p>
                  </div>
                )}
            </div>
          ) : (
            <div className="flex-1 flex items-center justify-center p-4">
              <p className="text-sm text-text-muted text-center">
                Selecciona una pantalla para ver su analisis
              </p>
            </div>
          )}
        </div>
      </div>
      )}

      {/* Add Pantalla Modal */}
      {showAddModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-bg-secondary rounded-xl border border-border w-full max-w-md mx-4 overflow-hidden">
            <div className="flex items-center justify-between p-4 border-b border-border">
              <h3 className="text-lg font-semibold text-text-primary">Agregar pantalla</h3>
              <button
                onClick={() => setShowAddModal(false)}
                className="p-1.5 text-text-muted hover:text-text-primary hover:bg-bg-tertiary rounded-lg transition-colors"
              >
                <X size={18} />
              </button>
            </div>
            <div className="p-4">
              {/* Upload area */}
              <div
                onDrop={handleDrop}
                onDragOver={(e) => e.preventDefault()}
                className="border-2 border-dashed border-border hover:border-primary rounded-xl p-8 text-center transition-colors cursor-pointer"
                onClick={() => document.getElementById('file-upload')?.click()}
              >
                {uploading ? (
                  <Loader2 size={32} className="mx-auto animate-spin text-primary mb-2" />
                ) : (
                  <Upload size={32} className="mx-auto text-text-muted mb-2" />
                )}
                <p className="text-text-primary font-medium mb-1">
                  {uploading ? 'Subiendo...' : 'Arrastra una imagen aqui'}
                </p>
                <p className="text-sm text-text-muted">
                  o haz click para seleccionar
                </p>
                <input
                  id="file-upload"
                  type="file"
                  accept="image/*"
                  multiple
                  onChange={(e) => handleFileUpload(e.target.files)}
                  className="hidden"
                />
              </div>

              {/* URL option - disabled for now */}
              <div className="mt-4 pt-4 border-t border-border">
                <p className="text-sm text-text-muted text-center">
                  La captura desde URL estara disponible proximamente
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
