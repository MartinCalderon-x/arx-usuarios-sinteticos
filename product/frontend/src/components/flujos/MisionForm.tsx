import { useState, useEffect } from 'react';
import { X, Target, Play, MousePointer2 } from 'lucide-react';
import type { Mision, MisionCreate, Pantalla, ElementoClickeable } from '../../lib/api';

interface MisionFormProps {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (data: MisionCreate) => Promise<void>;
  pantallas: Pantalla[];
  mision?: Mision | null;
}

export function MisionForm({ isOpen, onClose, onSubmit, pantallas, mision }: MisionFormProps) {
  const [nombre, setNombre] = useState('');
  const [instrucciones, setInstrucciones] = useState('');
  const [pantallaInicioId, setPantallaInicioId] = useState<string>('');
  const [pantallaObjetivoId, setPantallaObjetivoId] = useState<string>('');
  const [elementoObjetivo, setElementoObjetivo] = useState<{ tipo?: string; texto?: string }>({});
  const [maxPasos, setMaxPasos] = useState(10);
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Get elementos from selected pantalla objetivo
  const elementosDisponibles: ElementoClickeable[] =
    pantallas.find((p) => p.id === pantallaObjetivoId)?.elementos_clickeables || [];

  useEffect(() => {
    if (mision) {
      setNombre(mision.nombre);
      setInstrucciones(mision.instrucciones);
      setPantallaInicioId(mision.pantalla_inicio_id || '');
      setPantallaObjetivoId(mision.pantalla_objetivo_id || '');
      setElementoObjetivo(mision.elemento_objetivo || {});
      setMaxPasos(mision.max_pasos);
    } else {
      // Set defaults
      setNombre('');
      setInstrucciones('');
      setPantallaInicioId(pantallas[0]?.id || '');
      setPantallaObjetivoId('');
      setElementoObjetivo({});
      setMaxPasos(10);
    }
  }, [mision, pantallas, isOpen]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!nombre.trim() || !instrucciones.trim()) return;

    setIsSubmitting(true);
    try {
      await onSubmit({
        nombre: nombre.trim(),
        instrucciones: instrucciones.trim(),
        pantalla_inicio_id: pantallaInicioId || undefined,
        pantalla_objetivo_id: pantallaObjetivoId || undefined,
        elemento_objetivo: elementoObjetivo.texto ? elementoObjetivo : undefined,
        max_pasos: maxPasos,
      });
      onClose();
    } catch (error) {
      console.error('Error saving mision:', error);
    } finally {
      setIsSubmitting(false);
    }
  }

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-bg-secondary rounded-xl border border-border w-full max-w-lg mx-4 overflow-hidden max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-border flex-shrink-0">
          <div className="flex items-center gap-2">
            <Target size={20} className="text-primary" />
            <h3 className="text-lg font-semibold text-text-primary">
              {mision ? 'Editar Mision' : 'Nueva Mision'}
            </h3>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 text-text-muted hover:text-text-primary hover:bg-bg-tertiary rounded-lg transition-colors"
          >
            <X size={18} />
          </button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="flex-1 overflow-y-auto">
          <div className="p-4 space-y-4">
            {/* Nombre */}
            <div>
              <label className="block text-sm font-medium text-text-primary mb-1">
                Nombre de la mision
              </label>
              <input
                type="text"
                value={nombre}
                onChange={(e) => setNombre(e.target.value)}
                placeholder="Ej: Encontrar boton de contacto"
                className="w-full px-3 py-2 bg-bg-tertiary border border-border rounded-lg text-text-primary placeholder:text-text-muted focus:outline-none focus:border-primary"
                required
              />
            </div>

            {/* Instrucciones */}
            <div>
              <label className="block text-sm font-medium text-text-primary mb-1">
                Instrucciones para el usuario
              </label>
              <textarea
                value={instrucciones}
                onChange={(e) => setInstrucciones(e.target.value)}
                placeholder="Describe la tarea que el usuario debe completar..."
                rows={3}
                className="w-full px-3 py-2 bg-bg-tertiary border border-border rounded-lg text-text-primary placeholder:text-text-muted focus:outline-none focus:border-primary resize-none"
                required
              />
            </div>

            {/* Pantalla inicio */}
            <div>
              <label className="block text-sm font-medium text-text-primary mb-1">
                Pantalla de inicio
              </label>
              <select
                value={pantallaInicioId}
                onChange={(e) => setPantallaInicioId(e.target.value)}
                className="w-full px-3 py-2 bg-bg-tertiary border border-border rounded-lg text-text-primary focus:outline-none focus:border-primary"
              >
                {pantallas.map((p, idx) => (
                  <option key={p.id} value={p.id}>
                    {p.titulo || `Pantalla ${idx + 1}`}
                  </option>
                ))}
              </select>
            </div>

            {/* Pantalla objetivo */}
            <div>
              <label className="block text-sm font-medium text-text-primary mb-1">
                Pantalla objetivo (opcional)
              </label>
              <select
                value={pantallaObjetivoId}
                onChange={(e) => {
                  setPantallaObjetivoId(e.target.value);
                  setElementoObjetivo({});
                }}
                className="w-full px-3 py-2 bg-bg-tertiary border border-border rounded-lg text-text-primary focus:outline-none focus:border-primary"
              >
                <option value="">Seleccionar pantalla...</option>
                {pantallas.map((p, idx) => (
                  <option key={p.id} value={p.id}>
                    {p.titulo || `Pantalla ${idx + 1}`}
                  </option>
                ))}
              </select>
            </div>

            {/* Elemento objetivo */}
            {pantallaObjetivoId && elementosDisponibles.length > 0 && (
              <div>
                <label className="block text-sm font-medium text-text-primary mb-1">
                  <MousePointer2 size={14} className="inline mr-1" />
                  Elemento objetivo
                </label>
                <select
                  value={elementoObjetivo.texto || ''}
                  onChange={(e) => {
                    const elem = elementosDisponibles.find((el) => el.texto === e.target.value);
                    setElementoObjetivo(
                      elem ? { tipo: elem.tipo, texto: elem.texto } : {}
                    );
                  }}
                  className="w-full px-3 py-2 bg-bg-tertiary border border-border rounded-lg text-text-primary focus:outline-none focus:border-primary"
                >
                  <option value="">Seleccionar elemento...</option>
                  {elementosDisponibles.map((elem) => (
                    <option key={elem.id} value={elem.texto}>
                      [{elem.tipo}] {elem.texto || elem.descripcion || 'Sin texto'}
                    </option>
                  ))}
                </select>
              </div>
            )}

            {pantallaObjetivoId && elementosDisponibles.length === 0 && (
              <div className="p-3 bg-warning/10 border border-warning/30 rounded-lg">
                <p className="text-sm text-warning">
                  Esta pantalla no tiene elementos detectados. Usa "Detectar elementos" primero.
                </p>
              </div>
            )}

            {/* Elemento objetivo manual */}
            <div>
              <label className="block text-sm font-medium text-text-primary mb-1">
                O escribe el texto del elemento objetivo
              </label>
              <input
                type="text"
                value={elementoObjetivo.texto || ''}
                onChange={(e) =>
                  setElementoObjetivo({ ...elementoObjetivo, texto: e.target.value })
                }
                placeholder="Ej: Contacto, Enviar, Comprar"
                className="w-full px-3 py-2 bg-bg-tertiary border border-border rounded-lg text-text-primary placeholder:text-text-muted focus:outline-none focus:border-primary"
              />
            </div>

            {/* Max pasos */}
            <div>
              <label className="block text-sm font-medium text-text-primary mb-1">
                Pasos maximos
              </label>
              <input
                type="number"
                value={maxPasos}
                onChange={(e) => setMaxPasos(parseInt(e.target.value) || 10)}
                min={1}
                max={50}
                className="w-full px-3 py-2 bg-bg-tertiary border border-border rounded-lg text-text-primary focus:outline-none focus:border-primary"
              />
              <p className="text-xs text-text-muted mt-1">
                La simulacion fallara si excede este numero de pasos
              </p>
            </div>
          </div>

          {/* Footer */}
          <div className="p-4 border-t border-border flex justify-end gap-2 flex-shrink-0">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-text-secondary hover:text-text-primary hover:bg-bg-tertiary rounded-lg transition-colors"
            >
              Cancelar
            </button>
            <button
              type="submit"
              disabled={isSubmitting || !nombre.trim() || !instrucciones.trim()}
              className="flex items-center gap-2 px-4 py-2 bg-primary hover:bg-primary-dark text-white rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isSubmitting ? (
                'Guardando...'
              ) : (
                <>
                  <Play size={16} />
                  {mision ? 'Guardar cambios' : 'Crear mision'}
                </>
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
