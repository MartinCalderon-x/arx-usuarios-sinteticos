import { useState, useEffect } from 'react';
import {
  Play,
  Pause,
  SkipBack,
  SkipForward,
  ChevronLeft,
  ChevronRight,
  MousePointer2,
  AlertTriangle,
  CheckCircle2,
  XCircle,
} from 'lucide-react';
import type { Simulacion, SimulacionPaso } from '../../lib/api';

interface SimulationPlayerProps {
  simulacion: Simulacion;
  pantallaImagenes: Record<string, string>;
}

export function SimulationPlayer({ simulacion, pantallaImagenes }: SimulationPlayerProps) {
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playSpeed, setPlaySpeed] = useState(1000);

  const path = simulacion.path_tomado || [];
  const currentPaso: SimulacionPaso | undefined = path[currentStep];

  // Auto-play
  useEffect(() => {
    if (!isPlaying || currentStep >= path.length - 1) {
      setIsPlaying(false);
      return;
    }

    const timer = setTimeout(() => {
      setCurrentStep((prev) => prev + 1);
    }, playSpeed);

    return () => clearTimeout(timer);
  }, [isPlaying, currentStep, path.length, playSpeed]);

  function goToStep(step: number) {
    setCurrentStep(Math.max(0, Math.min(step, path.length - 1)));
  }

  if (path.length === 0) {
    return (
      <div className="text-center py-8 text-text-muted">
        <p>No hay pasos para mostrar</p>
      </div>
    );
  }

  const currentImageUrl = currentPaso ? pantallaImagenes[currentPaso.pantalla_id] : undefined;

  return (
    <div className="bg-bg-secondary rounded-xl border border-border overflow-hidden">
      {/* Header */}
      <div className="p-4 border-b border-border">
        <div className="flex items-center justify-between mb-2">
          <h4 className="font-medium text-text-primary">
            Reproductor de Simulacion
          </h4>
          <div className="flex items-center gap-2">
            {simulacion.exito ? (
              <span className="flex items-center gap-1 px-2 py-0.5 bg-success/10 text-success text-xs rounded">
                <CheckCircle2 size={12} />
                Exito
              </span>
            ) : (
              <span className="flex items-center gap-1 px-2 py-0.5 bg-error/10 text-error text-xs rounded">
                <XCircle size={12} />
                Fallida
              </span>
            )}
          </div>
        </div>

        {/* Stats */}
        <div className="flex gap-4 text-sm text-text-secondary">
          <span>Pasos: {simulacion.pasos_totales}</span>
          <span>Misclicks: {simulacion.misclicks}</span>
          <span>
            Tiempo: {Math.round((simulacion.tiempo_estimado_ms || 0) / 1000)}s
          </span>
        </div>
      </div>

      {/* Player area */}
      <div className="flex">
        {/* Image/Screen view */}
        <div className="flex-1 p-4 bg-bg-tertiary/30 min-h-[300px] flex items-center justify-center">
          {currentImageUrl ? (
            <img
              src={currentImageUrl}
              alt={currentPaso?.pantalla_titulo || 'Pantalla'}
              className="max-w-full max-h-[400px] object-contain rounded-lg shadow-lg"
            />
          ) : (
            <div className="text-center text-text-muted">
              <p>Imagen no disponible</p>
            </div>
          )}
        </div>

        {/* Step details */}
        <div className="w-72 border-l border-border p-4 flex flex-col">
          <h5 className="text-sm font-medium text-text-muted uppercase tracking-wide mb-3">
            Paso {currentStep + 1} de {path.length}
          </h5>

          {currentPaso && (
            <div className="space-y-3 flex-1">
              {/* Element clicked */}
              <div>
                <span className="text-xs text-text-muted">Elemento clickeado</span>
                <div className="flex items-center gap-2 mt-1">
                  <MousePointer2 size={14} className="text-primary" />
                  <span className="text-sm text-text-primary font-medium">
                    {currentPaso.elemento_clickeado.texto || 'Sin texto'}
                  </span>
                </div>
              </div>

              {/* Razonamiento */}
              <div>
                <span className="text-xs text-text-muted">Razonamiento</span>
                <p className="text-sm text-text-secondary mt-1">
                  "{currentPaso.razonamiento}"
                </p>
              </div>

              {/* Emotion */}
              <div className="flex items-center gap-2">
                <span className="text-xs text-text-muted">Emocion:</span>
                <span
                  className={`text-xs px-2 py-0.5 rounded ${
                    currentPaso.emocion === 'frustrado'
                      ? 'bg-error/10 text-error'
                      : currentPaso.emocion === 'confundido'
                      ? 'bg-warning/10 text-warning'
                      : currentPaso.emocion === 'satisfecho'
                      ? 'bg-success/10 text-success'
                      : 'bg-bg-tertiary text-text-secondary'
                  }`}
                >
                  {currentPaso.emocion}
                </span>
              </div>

              {/* Warnings */}
              {currentPaso.es_misclick && (
                <div className="flex items-center gap-2 p-2 bg-warning/10 border border-warning/30 rounded-lg">
                  <AlertTriangle size={14} className="text-warning" />
                  <span className="text-xs text-warning">Misclick</span>
                </div>
              )}

              {currentPaso.confusion && (
                <div className="flex items-center gap-2 p-2 bg-orange-500/10 border border-orange-500/30 rounded-lg">
                  <AlertTriangle size={14} className="text-orange-500" />
                  <span className="text-xs text-orange-500">Usuario confundido</span>
                </div>
              )}

              {/* Time */}
              <div className="text-xs text-text-muted">
                Tiempo en este paso: {Math.round(currentPaso.tiempo_ms / 1000)}s
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Controls */}
      <div className="p-4 border-t border-border">
        {/* Progress bar */}
        <div className="flex items-center gap-2 mb-3">
          {path.map((paso, idx) => (
            <button
              key={idx}
              onClick={() => goToStep(idx)}
              className={`flex-1 h-1.5 rounded-full transition-colors ${
                idx === currentStep
                  ? 'bg-primary'
                  : idx < currentStep
                  ? paso.es_misclick
                    ? 'bg-warning'
                    : 'bg-success'
                  : 'bg-bg-tertiary'
              }`}
              title={`Paso ${idx + 1}: ${paso.elemento_clickeado.texto}`}
            />
          ))}
        </div>

        {/* Playback controls */}
        <div className="flex items-center justify-center gap-2">
          <button
            onClick={() => goToStep(0)}
            className="p-2 text-text-muted hover:text-text-primary hover:bg-bg-tertiary rounded-lg transition-colors"
            title="Ir al inicio"
          >
            <SkipBack size={18} />
          </button>

          <button
            onClick={() => goToStep(currentStep - 1)}
            disabled={currentStep === 0}
            className="p-2 text-text-muted hover:text-text-primary hover:bg-bg-tertiary rounded-lg transition-colors disabled:opacity-30"
            title="Paso anterior"
          >
            <ChevronLeft size={18} />
          </button>

          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className="p-3 bg-primary hover:bg-primary-dark text-white rounded-full transition-colors"
            title={isPlaying ? 'Pausar' : 'Reproducir'}
          >
            {isPlaying ? <Pause size={20} /> : <Play size={20} />}
          </button>

          <button
            onClick={() => goToStep(currentStep + 1)}
            disabled={currentStep >= path.length - 1}
            className="p-2 text-text-muted hover:text-text-primary hover:bg-bg-tertiary rounded-lg transition-colors disabled:opacity-30"
            title="Siguiente paso"
          >
            <ChevronRight size={18} />
          </button>

          <button
            onClick={() => goToStep(path.length - 1)}
            className="p-2 text-text-muted hover:text-text-primary hover:bg-bg-tertiary rounded-lg transition-colors"
            title="Ir al final"
          >
            <SkipForward size={18} />
          </button>

          {/* Speed control */}
          <div className="ml-4 flex items-center gap-2">
            <span className="text-xs text-text-muted">Velocidad:</span>
            <select
              value={playSpeed}
              onChange={(e) => setPlaySpeed(parseInt(e.target.value))}
              className="px-2 py-1 bg-bg-tertiary border border-border rounded text-sm text-text-primary"
            >
              <option value={2000}>0.5x</option>
              <option value={1000}>1x</option>
              <option value={500}>2x</option>
              <option value={250}>4x</option>
            </select>
          </div>
        </div>
      </div>
    </div>
  );
}
