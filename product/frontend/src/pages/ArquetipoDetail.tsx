import { useEffect, useState } from 'react';
import { useNavigate, useParams, Link } from 'react-router-dom';
import { ArrowLeft, Edit, Copy, Trash2, Loader2, User, Target, AlertTriangle, Briefcase, Monitor, Building } from 'lucide-react';
import { arquetiposApi, type Arquetipo } from '../lib/api';

const NIVEL_COLORS: Record<string, string> = {
  bajo: 'bg-error/10 text-error border-error/20',
  medio: 'bg-warning/10 text-warning border-warning/20',
  alto: 'bg-success/10 text-success border-success/20',
};


export function ArquetipoDetail() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [arquetipo, setArquetipo] = useState<Arquetipo | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (id) loadArquetipo(id);
  }, [id]);

  async function loadArquetipo(id: string) {
    try {
      const data = await arquetiposApi.get(id);
      setArquetipo(data);
    } catch (error) {
      console.error('Error loading arquetipo:', error);
    } finally {
      setLoading(false);
    }
  }

  async function handleDelete() {
    if (!id || !confirm('Estas seguro de eliminar este arquetipo?')) return;
    try {
      await arquetiposApi.delete(id);
      navigate('/arquetipos');
    } catch (error) {
      console.error('Error deleting arquetipo:', error);
    }
  }

  async function handleDuplicate() {
    if (!arquetipo) return;
    try {
      const { id, created_at, ...rest } = arquetipo;
      const newArquetipo = await arquetiposApi.create({
        ...rest,
        nombre: `${rest.nombre} (copia)`,
        descripcion: rest.descripcion || '',
      });
      navigate(`/arquetipos/${newArquetipo.id}`);
    } catch (error) {
      console.error('Error duplicating arquetipo:', error);
      alert('Error al duplicar el arquetipo');
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="w-8 h-8 animate-spin text-primary" />
      </div>
    );
  }

  if (!arquetipo) {
    return (
      <div className="text-center py-12">
        <User size={48} className="mx-auto text-text-muted mb-4" />
        <h3 className="text-lg font-medium text-text-primary mb-2">Arquetipo no encontrado</h3>
        <Link to="/arquetipos" className="text-primary hover:underline">
          Volver a arquetipos
        </Link>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-4">
          <button
            onClick={() => navigate('/arquetipos')}
            className="p-2 rounded-lg hover:bg-bg-secondary text-text-secondary transition-colors"
          >
            <ArrowLeft size={20} />
          </button>
          <div className="flex items-center gap-4">
            <div className="p-3 bg-primary/10 rounded-xl">
              <User size={28} className="text-primary" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-text-primary">{arquetipo.nombre}</h1>
              <p className="text-text-secondary mt-1">{arquetipo.descripcion}</p>
            </div>
          </div>
        </div>
        <div className="flex gap-2">
          <button
            onClick={handleDuplicate}
            className="p-2 text-text-muted hover:text-secondary hover:bg-bg-secondary rounded-lg transition-colors"
            title="Duplicar"
          >
            <Copy size={18} />
          </button>
          <Link
            to={`/arquetipos/${arquetipo.id}/editar`}
            className="p-2 text-text-muted hover:text-primary hover:bg-bg-secondary rounded-lg transition-colors"
            title="Editar"
          >
            <Edit size={18} />
          </Link>
          <button
            onClick={handleDelete}
            className="p-2 text-text-muted hover:text-error hover:bg-error/10 rounded-lg transition-colors"
            title="Eliminar"
          >
            <Trash2 size={18} />
          </button>
        </div>
      </div>

      {/* Quick stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {arquetipo.edad && (
          <div className="bg-bg-secondary rounded-xl p-4 border border-border">
            <p className="text-text-muted text-sm mb-1">Edad</p>
            <p className="text-xl font-semibold text-text-primary">{arquetipo.edad} años</p>
          </div>
        )}
        {arquetipo.genero && (
          <div className="bg-bg-secondary rounded-xl p-4 border border-border">
            <p className="text-text-muted text-sm mb-1">Género</p>
            <p className="text-xl font-semibold text-text-primary capitalize">{arquetipo.genero}</p>
          </div>
        )}
        {arquetipo.nivel_digital && (
          <div className={`rounded-xl p-4 border ${NIVEL_COLORS[arquetipo.nivel_digital] || 'bg-bg-secondary border-border'}`}>
            <div className="flex items-center gap-2 mb-1">
              <Monitor size={14} />
              <p className="text-sm">Nivel Digital</p>
            </div>
            <p className="text-xl font-semibold capitalize">{arquetipo.nivel_digital}</p>
          </div>
        )}
        {arquetipo.industria && (
          <div className="bg-secondary/10 text-secondary rounded-xl p-4 border border-secondary/20">
            <div className="flex items-center gap-2 mb-1">
              <Building size={14} />
              <p className="text-sm">Industria</p>
            </div>
            <p className="text-xl font-semibold capitalize">{arquetipo.industria}</p>
          </div>
        )}
      </div>

      {/* Details */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Left column */}
        <div className="space-y-6">
          {arquetipo.ocupacion && (
            <div className="bg-bg-secondary rounded-xl p-5 border border-border">
              <div className="flex items-center gap-2 mb-3">
                <Briefcase size={18} className="text-primary" />
                <h3 className="font-medium text-text-primary">Ocupación</h3>
              </div>
              <p className="text-text-secondary">{arquetipo.ocupacion}</p>
            </div>
          )}

          {arquetipo.contexto && (
            <div className="bg-bg-secondary rounded-xl p-5 border border-border">
              <h3 className="font-medium text-text-primary mb-3">Contexto</h3>
              <p className="text-text-secondary">{arquetipo.contexto}</p>
            </div>
          )}

          {arquetipo.comportamiento && (
            <div className="bg-bg-secondary rounded-xl p-5 border border-border">
              <h3 className="font-medium text-text-primary mb-3">Comportamiento</h3>
              <p className="text-text-secondary">{arquetipo.comportamiento}</p>
            </div>
          )}
        </div>

        {/* Right column */}
        <div className="space-y-6">
          {arquetipo.frustraciones && arquetipo.frustraciones.length > 0 && (
            <div className="bg-bg-secondary rounded-xl p-5 border border-border">
              <div className="flex items-center gap-2 mb-4">
                <AlertTriangle size={18} className="text-error" />
                <h3 className="font-medium text-text-primary">Frustraciones</h3>
              </div>
              <ul className="space-y-2">
                {arquetipo.frustraciones.map((f, i) => (
                  <li key={i} className="flex items-start gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-error mt-2 flex-shrink-0" />
                    <span className="text-text-secondary">{f}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {arquetipo.objetivos && arquetipo.objetivos.length > 0 && (
            <div className="bg-bg-secondary rounded-xl p-5 border border-border">
              <div className="flex items-center gap-2 mb-4">
                <Target size={18} className="text-success" />
                <h3 className="font-medium text-text-primary">Objetivos</h3>
              </div>
              <ul className="space-y-2">
                {arquetipo.objetivos.map((o, i) => (
                  <li key={i} className="flex items-start gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-success mt-2 flex-shrink-0" />
                    <span className="text-text-secondary">{o}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>

      {/* Actions */}
      <div className="flex justify-center gap-4 pt-4">
        <Link
          to={`/interaccion?arquetipo=${arquetipo.id}`}
          className="px-6 py-2.5 bg-primary hover:bg-primary-dark text-white rounded-lg transition-colors"
        >
          Iniciar conversación
        </Link>
        <Link
          to={`/arquetipos/${arquetipo.id}/editar`}
          className="px-6 py-2.5 bg-bg-secondary hover:bg-bg-tertiary text-text-primary border border-border rounded-lg transition-colors"
        >
          Editar arquetipo
        </Link>
      </div>
    </div>
  );
}
