import { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Plus, Layers, Trash2, Edit, Search, Eye, Archive, ArchiveRestore } from 'lucide-react';
import { flujosApi, type Flujo } from '../lib/api';

const ESTADO_COLORS: Record<string, string> = {
  activo: 'bg-success/10 text-success',
  archivado: 'bg-text-muted/10 text-text-muted',
};

export function Flujos() {
  const navigate = useNavigate();
  const [flujos, setFlujos] = useState<Flujo[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [filterEstado, setFilterEstado] = useState('');

  useEffect(() => {
    loadFlujos();
  }, []);

  async function loadFlujos() {
    try {
      const { flujos } = await flujosApi.list();
      setFlujos(flujos);
    } catch (error) {
      console.error('Error loading flujos:', error);
    } finally {
      setLoading(false);
    }
  }

  async function handleDelete(id: string) {
    if (!confirm('Estas seguro de eliminar este flujo y todas sus pantallas?')) return;
    try {
      await flujosApi.delete(id);
      setFlujos(flujos.filter(f => f.id !== id));
    } catch (error) {
      console.error('Error deleting flujo:', error);
    }
  }

  async function handleToggleArchive(flujo: Flujo) {
    const newEstado = flujo.estado === 'activo' ? 'archivado' : 'activo';
    try {
      await flujosApi.update(flujo.id, { estado: newEstado });
      setFlujos(flujos.map(f => f.id === flujo.id ? { ...f, estado: newEstado } : f));
    } catch (error) {
      console.error('Error updating flujo:', error);
    }
  }

  const filtered = flujos.filter(f => {
    const matchesSearch =
      f.nombre.toLowerCase().includes(search.toLowerCase()) ||
      f.descripcion?.toLowerCase().includes(search.toLowerCase());
    const matchesEstado = !filterEstado || f.estado === filterEstado;
    return matchesSearch && matchesEstado;
  });

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-text-primary">Flujos</h1>
          <p className="text-text-secondary mt-1">Analiza journeys y flujos de usuario completos</p>
        </div>
        <Link
          to="/flujos/nuevo"
          className="flex items-center gap-2 px-4 py-2 bg-primary hover:bg-primary-dark text-white rounded-lg transition-colors"
        >
          <Plus size={18} />
          Nuevo flujo
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
            placeholder="Buscar flujos..."
            className="w-full pl-10 pr-4 py-2.5 rounded-lg border border-border bg-bg-primary text-text-primary placeholder:text-text-muted focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary"
          />
        </div>
        <select
          value={filterEstado}
          onChange={(e) => setFilterEstado(e.target.value)}
          className="px-4 py-2.5 rounded-lg border border-border bg-bg-primary text-text-primary focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary"
        >
          <option value="">Todos los estados</option>
          <option value="activo">Activos</option>
          <option value="archivado">Archivados</option>
        </select>
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
          <Layers size={48} className="mx-auto text-text-muted mb-4" />
          <h3 className="text-lg font-medium text-text-primary mb-2">
            {search || filterEstado ? 'No se encontraron flujos' : 'No hay flujos'}
          </h3>
          <p className="text-text-secondary mb-4">
            {search || filterEstado ? 'Intenta con otra busqueda' : 'Crea tu primer flujo para comenzar'}
          </p>
          {!search && !filterEstado && (
            <Link
              to="/flujos/nuevo"
              className="inline-flex items-center gap-2 px-4 py-2 bg-primary hover:bg-primary-dark text-white rounded-lg transition-colors"
            >
              <Plus size={18} />
              Crear flujo
            </Link>
          )}
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filtered.map(flujo => (
            <div
              key={flujo.id}
              className="bg-bg-secondary rounded-xl p-6 border border-border hover:border-border-dark transition-colors group"
            >
              <div className="flex items-start justify-between mb-3">
                <div className="p-2 bg-primary/10 rounded-lg">
                  <Layers size={20} className="text-primary" />
                </div>
                <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                  <button
                    onClick={() => navigate(`/flujos/${flujo.id}`)}
                    className="p-2 text-text-muted hover:text-primary hover:bg-bg-tertiary rounded-lg transition-colors"
                    title="Ver detalle"
                  >
                    <Eye size={16} />
                  </button>
                  <button
                    onClick={() => handleToggleArchive(flujo)}
                    className="p-2 text-text-muted hover:text-secondary hover:bg-bg-tertiary rounded-lg transition-colors"
                    title={flujo.estado === 'activo' ? 'Archivar' : 'Restaurar'}
                  >
                    {flujo.estado === 'activo' ? <Archive size={16} /> : <ArchiveRestore size={16} />}
                  </button>
                  <Link
                    to={`/flujos/${flujo.id}/editar`}
                    className="p-2 text-text-muted hover:text-primary hover:bg-bg-tertiary rounded-lg transition-colors"
                    title="Editar"
                  >
                    <Edit size={16} />
                  </Link>
                  <button
                    onClick={() => handleDelete(flujo.id)}
                    className="p-2 text-text-muted hover:text-error hover:bg-error/10 rounded-lg transition-colors"
                    title="Eliminar"
                  >
                    <Trash2 size={16} />
                  </button>
                </div>
              </div>
              <h3 className="font-semibold text-text-primary mb-1">{flujo.nombre}</h3>
              <p className="text-sm text-text-secondary line-clamp-2 mb-3">
                {flujo.descripcion || 'Sin descripcion'}
              </p>
              <div className="flex flex-wrap gap-2 text-xs">
                <span className={`px-2 py-1 rounded capitalize ${ESTADO_COLORS[flujo.estado] || 'bg-bg-tertiary text-text-secondary'}`}>
                  {flujo.estado}
                </span>
                <span className="px-2 py-1 bg-bg-tertiary text-text-secondary rounded">
                  {flujo.total_pantallas} {flujo.total_pantallas === 1 ? 'pantalla' : 'pantallas'}
                </span>
                {flujo.url_inicial && (
                  <span className="px-2 py-1 bg-secondary/10 text-secondary rounded truncate max-w-[150px]">
                    {new URL(flujo.url_inicial).hostname}
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
