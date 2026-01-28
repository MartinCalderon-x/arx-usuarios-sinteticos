import { useState } from 'react';
import { MousePointer2, Eye, EyeOff } from 'lucide-react';
import type { ElementoClickeable } from '../../lib/api';

interface ElementOverlayProps {
  imageUrl: string;
  elementos: ElementoClickeable[];
  onElementClick?: (elemento: ElementoClickeable) => void;
  selectedElementId?: string;
  showLabels?: boolean;
}

const ELEMENT_COLORS: Record<string, string> = {
  button: 'border-blue-500 bg-blue-500/20',
  link: 'border-green-500 bg-green-500/20',
  input: 'border-purple-500 bg-purple-500/20',
  tab: 'border-yellow-500 bg-yellow-500/20',
  menu: 'border-orange-500 bg-orange-500/20',
  icon: 'border-pink-500 bg-pink-500/20',
  card: 'border-cyan-500 bg-cyan-500/20',
  image: 'border-indigo-500 bg-indigo-500/20',
};

const ELEMENT_LABEL_COLORS: Record<string, string> = {
  button: 'bg-blue-500',
  link: 'bg-green-500',
  input: 'bg-purple-500',
  tab: 'bg-yellow-500',
  menu: 'bg-orange-500',
  icon: 'bg-pink-500',
  card: 'bg-cyan-500',
  image: 'bg-indigo-500',
};

export function ElementOverlay({
  imageUrl,
  elementos,
  onElementClick,
  selectedElementId,
  showLabels = true,
}: ElementOverlayProps) {
  const [showOverlay, setShowOverlay] = useState(true);
  const [hoveredElement, setHoveredElement] = useState<string | null>(null);

  return (
    <div className="relative inline-block">
      {/* Toggle overlay button */}
      <button
        onClick={() => setShowOverlay(!showOverlay)}
        className="absolute top-2 right-2 z-20 p-2 bg-bg-secondary/90 backdrop-blur rounded-lg border border-border hover:bg-bg-tertiary transition-colors"
        title={showOverlay ? 'Ocultar elementos' : 'Mostrar elementos'}
      >
        {showOverlay ? <Eye size={16} /> : <EyeOff size={16} />}
      </button>

      {/* Image */}
      <img
        src={imageUrl}
        alt="Screenshot"
        className="max-w-full max-h-full object-contain"
      />

      {/* Elements overlay */}
      {showOverlay && (
        <div className="absolute inset-0">
          {elementos.map((elemento) => {
            const { bbox } = elemento;
            const isSelected = elemento.id === selectedElementId;
            const isHovered = elemento.id === hoveredElement;
            const colorClass = ELEMENT_COLORS[elemento.tipo] || 'border-gray-500 bg-gray-500/20';
            const labelColor = ELEMENT_LABEL_COLORS[elemento.tipo] || 'bg-gray-500';

            return (
              <div
                key={elemento.id}
                className={`absolute border-2 rounded cursor-pointer transition-all ${colorClass} ${
                  isSelected ? 'ring-2 ring-primary ring-offset-2' : ''
                } ${isHovered ? 'border-opacity-100' : 'border-opacity-70'}`}
                style={{
                  left: `${bbox.x}%`,
                  top: `${bbox.y}%`,
                  width: `${bbox.width}%`,
                  height: `${bbox.height}%`,
                }}
                onClick={() => onElementClick?.(elemento)}
                onMouseEnter={() => setHoveredElement(elemento.id)}
                onMouseLeave={() => setHoveredElement(null)}
              >
                {/* Label */}
                {showLabels && (isHovered || isSelected) && (
                  <div
                    className={`absolute -top-6 left-0 px-2 py-0.5 text-xs text-white rounded whitespace-nowrap ${labelColor}`}
                  >
                    <MousePointer2 size={10} className="inline mr-1" />
                    {elemento.texto || elemento.tipo}
                  </div>
                )}

                {/* CTA indicator */}
                {elemento.es_cta_principal && (
                  <div className="absolute -top-1 -right-1 w-3 h-3 bg-primary rounded-full border border-white" />
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* Legend */}
      {showOverlay && elementos.length > 0 && (
        <div className="absolute bottom-2 left-2 flex flex-wrap gap-2 max-w-[60%]">
          {Object.entries(ELEMENT_LABEL_COLORS)
            .filter(([tipo]) => elementos.some((e) => e.tipo === tipo))
            .map(([tipo, color]) => (
              <span
                key={tipo}
                className={`px-2 py-0.5 text-xs text-white rounded ${color}`}
              >
                {tipo}
              </span>
            ))}
        </div>
      )}
    </div>
  );
}
