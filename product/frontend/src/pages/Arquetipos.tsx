import { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Plus, User, Trash2, Edit, Search, Copy, Eye } from 'lucide-react';
import { arquetiposApi, type Arquetipo } from '../lib/api';

const NIVEL_COLORS: Record<string, string> = {
  bajo: 'bg-error/10 text-error',
  medio: 'bg-warning/10 text-warning',
  alto: 'bg-success/10 text-success',
};

export function Arquetipos() {
  const navigate = useNavigate();
  const [arquetipos, setArquetipos] = useState<Arquetipo[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [filterIndustria, setFilterIndustria] = useState('');

  useEffect(() => {
    loadArquetipos();
  }, []);

  async function loadArquetipos() {
    try {
      const { arquetipos } = await arquetiposApi.list();
      setArquetipos(arquetipos);
    } catch (error) {
      console.error('Error loading arquetipos:', error);
    } finally {
      setLoading(false);
    }
  }

  async function handleDelete(id: string) {
    if (!confirm('Estas seguro de eliminar este arquetipo?')) return;
    try {
      await arquetiposApi.delete(id);
      setArquetipos(arquetipos.filter(a => a.id !== id));
    } catch (error) {
      console.error('Error deleting arquetipo:', error);
    }
  }

  async function handleDuplicate(arquetipo: Arquetipo) {
    try {
      const { id, created_at, ...rest } = arquetipo;
      await arquetiposApi.create({
        ...rest,
        nombre: `${rest.nombre} (copia)`,
        descripcion: rest.descripcion || '',
      });
      loadArquetipos();
    } catch (error) {
      console.error('Error duplicating arquetipo:', error);
      alert('Error al duplicar el arquetipo');
    }
  }

  // Get unique industries for filter
  const industrias = [...new Set(arquetipos.map(a => a.industria).filter(Boolean))];

  const filtered = arquetipos.filter(a => {
    const matchesSearch =
      a.nombre.toLowerCase().includes(search.toLowerCase()) ||
      a.descripcion?.toLowerCase().includes(search.toLowerCase());
    const matchesIndustria = !filterIndustria || a.industria === filterIndustria;
    return matchesSearch && matchesIndustria;
  });

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-text-primary">Arquetipos</h1>
          <p className="text-text-secondary mt-1">Gestiona tus usuarios sinteticos</p>
        </div>
        <Link
          to="/arquetipos/nuevo"
          className="flex items-center gap-2 px-4 py-2 bg-primary hover:bg-primary-dark text-white rounded-lg transition-colors"
        >
          <Plus size={18} />
          Nuevo arquetipo
        </Link>
      </div>

      {/* Filters */}
      <div className="flex flex-col sm:flex-row gap-3">
        <div className="relative flex-1">
          <Search size={18} className="absolute left-3 top-1/2 -translate-y-1/2 text-text-muted" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Buscar arquetipos..."
            className="w-full pl-10 pr-4 py-2.5 rounded-lg border border-border bg-bg-primary text-text-primary placeholder:text-text-muted focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary"
          />
        </div>
        {industrias.length > 0 && (
          <select
            value={filterIndustria}
            onChange={(e) => setFilterIndustria(e.target.value)}
            className="px-4 py-2.5 rounded-lg border border-border bg-bg-primary text-text-primary focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary"
          >
            <option value="">Todas las industrias</option>
            {industrias.map(ind => (
              <option key={ind} value={ind} className="capitalize">{ind}</option>
            ))}
          </select>
        )}
      </div>

      {/* List */}
      {loading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {[1, 2, 3].map(i => (
            <div key={i} className="bg-bg-secondary rounded-xl p-6 animate-pulse">
              <div className="h-6 bg-bg-tertiary rounded w-3/4 mb-3" />
              <div className="h-4 bg-bg-tertiary rounded w-full mb-2" />
              <div className="h-4 bg-bg-tertiary rounded w-2/3" />
            </div>
          ))}
        </div>
      ) : filtered.length === 0 ? (
        <div className="text-center py-12 bg-bg-secondary rounded-xl border border-border">
          <User size={48} className="mx-auto text-text-muted mb-4" />
          <h3 className="text-lg font-medium text-text-primary mb-2">
            {search || filterIndustria ? 'No se encontraron arquetipos' : 'No hay arquetipos'}
          </h3>
          <p className="text-text-secondary mb-4">
            {search || filterIndustria ? 'Intenta con otra busqueda' : 'Crea tu primer arquetipo para comenzar'}
          </p>
          {!search && !filterIndustria && (
            <Link
              to="/arquetipos/nuevo"
              className="inline-flex items-center gap-2 px-4 py-2 bg-primary hover:bg-primary-dark text-white rounded-lg transition-colors"
            >
              <Plus size={18} />
              Crear arquetipo
            </Link>
          )}
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filtered.map(arquetipo => (
            <div
              key={arquetipo.id}
              className="bg-bg-secondary rounded-xl p-6 border border-border hover:border-border-dark transition-colors group"
            >
              <div className="flex items-start justify-between mb-3">
                <div className="p-2 bg-primary/10 rounded-lg">
                  <User size={20} className="text-primary" />
                </div>
                <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                  <button
                    onClick={() => navigate(`/arquetipos/${arquetipo.id}`)}
                    className="p-2 text-text-muted hover:text-primary hover:bg-bg-tertiary rounded-lg transition-colors"
                    title="Ver detalle"
                  >
                    <Eye size={16} />
                  </button>
                  <button
                    onClick={() => handleDuplicate(arquetipo)}
                    className="p-2 text-text-muted hover:text-secondary hover:bg-bg-tertiary rounded-lg transition-colors"
                    title="Duplicar"
                  >
                    <Copy size={16} />
                  </button>
                  <Link
                    to={`/arquetipos/${arquetipo.id}/editar`}
                    className="p-2 text-text-muted hover:text-primary hover:bg-bg-tertiary rounded-lg transition-colors"
                    title="Editar"
                  >
                    <Edit size={16} />
                  </Link>
                  <button
                    onClick={() => handleDelete(arquetipo.id)}
                    className="p-2 text-text-muted hover:text-error hover:bg-error/10 rounded-lg transition-colors"
                    title="Eliminar"
                  >
                    <Trash2 size={16} />
                  </button>
                </div>
              </div>
              <h3 className="font-semibold text-text-primary mb-1">{arquetipo.nombre}</h3>
              <p className="text-sm text-text-secondary line-clamp-2 mb-3">
                {arquetipo.descripcion}
              </p>
              <div className="flex flex-wrap gap-2 text-xs">
                {arquetipo.edad && (
                  <span className="px-2 py-1 bg-bg-tertiary text-text-secondary rounded">
                    {arquetipo.edad} años
                  </span>
                )}
                {arquetipo.ocupacion && (
                  <span className="px-2 py-1 bg-bg-tertiary text-text-secondary rounded">
                    {arquetipo.ocupacion}
                  </span>
                )}
                {arquetipo.nivel_digital && (
                  <span className={`px-2 py-1 rounded capitalize ${NIVEL_COLORS[arquetipo.nivel_digital] || 'bg-bg-tertiary text-text-secondary'}`}>
                    {arquetipo.nivel_digital}
                  </span>
                )}
                {arquetipo.industria && (
                  <span className="px-2 py-1 bg-secondary/10 text-secondary rounded capitalize">
                    {arquetipo.industria}
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
