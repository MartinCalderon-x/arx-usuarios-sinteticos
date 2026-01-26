import { useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { ArrowLeft, Save, Loader2 } from 'lucide-react';
import { flujosApi, type FlujoCreate } from '../lib/api';

export function FlujoForm() {
  const { id } = useParams();
  const navigate = useNavigate();
  const isEditing = Boolean(id);

  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [form, setForm] = useState<FlujoCreate>({
    nombre: '',
    descripcion: '',
    url_inicial: '',
  });

  useEffect(() => {
    if (id) loadFlujo(id);
  }, [id]);

  async function loadFlujo(id: string) {
    setLoading(true);
    try {
      const flujo = await flujosApi.get(id);
      setForm({
        nombre: flujo.nombre,
        descripcion: flujo.descripcion || '',
        url_inicial: flujo.url_inicial || '',
      });
    } catch (error) {
      console.error('Error loading flujo:', error);
      navigate('/flujos');
    } finally {
      setLoading(false);
    }
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!form.nombre.trim()) return;

    setSaving(true);
    try {
      if (isEditing && id) {
        await flujosApi.update(id, form);
        navigate(`/flujos/${id}`);
      } else {
        const flujo = await flujosApi.create(form);
        navigate(`/flujos/${flujo.id}`);
      }
    } catch (error) {
      console.error('Error saving flujo:', error);
      alert('Error al guardar el flujo');
    } finally {
      setSaving(false);
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <Loader2 size={32} className="animate-spin text-primary" />
      </div>
    );
  }

  return (
    <div className="max-w-2xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center gap-4">
        <button
          onClick={() => navigate('/flujos')}
          className="p-2 text-text-secondary hover:text-text-primary hover:bg-bg-tertiary rounded-lg transition-colors"
        >
          <ArrowLeft size={20} />
        </button>
        <div>
          <h1 className="text-2xl font-bold text-text-primary">
            {isEditing ? 'Editar flujo' : 'Nuevo flujo'}
          </h1>
          <p className="text-text-secondary mt-1">
            {isEditing ? 'Modifica los detalles del flujo' : 'Crea un nuevo flujo de usuario'}
          </p>
        </div>
      </div>

      {/* Form */}
      <form onSubmit={handleSubmit} className="bg-bg-secondary rounded-xl p-6 border border-border space-y-6">
        <div>
          <label className="block text-sm font-medium text-text-primary mb-2">
            Nombre del flujo <span className="text-error">*</span>
          </label>
          <input
            type="text"
            value={form.nombre}
            onChange={(e) => setForm({ ...form, nombre: e.target.value })}
            placeholder="Ej: Onboarding de nuevos usuarios"
            className="w-full px-4 py-2.5 rounded-lg border border-border bg-bg-primary text-text-primary placeholder:text-text-muted focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary"
            required
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-text-primary mb-2">
            Descripcion
          </label>
          <textarea
            value={form.descripcion}
            onChange={(e) => setForm({ ...form, descripcion: e.target.value })}
            placeholder="Describe el proposito y contexto de este flujo..."
            rows={3}
            className="w-full px-4 py-2.5 rounded-lg border border-border bg-bg-primary text-text-primary placeholder:text-text-muted focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary resize-none"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-text-primary mb-2">
            URL inicial (opcional)
          </label>
          <input
            type="url"
            value={form.url_inicial}
            onChange={(e) => setForm({ ...form, url_inicial: e.target.value })}
            placeholder="https://ejemplo.com/inicio"
            className="w-full px-4 py-2.5 rounded-lg border border-border bg-bg-primary text-text-primary placeholder:text-text-muted focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary"
          />
          <p className="text-xs text-text-muted mt-1">
            Si agregas una URL, podras capturar pantallas automaticamente desde el sitio web
          </p>
        </div>

        {/* Actions */}
        <div className="flex items-center justify-end gap-3 pt-4 border-t border-border">
          <button
            type="button"
            onClick={() => navigate('/flujos')}
            className="px-4 py-2.5 text-text-secondary hover:text-text-primary hover:bg-bg-tertiary rounded-lg transition-colors"
          >
            Cancelar
          </button>
          <button
            type="submit"
            disabled={saving || !form.nombre.trim()}
            className="flex items-center gap-2 px-4 py-2.5 bg-primary hover:bg-primary-dark text-white rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {saving ? (
              <>
                <Loader2 size={18} className="animate-spin" />
                Guardando...
              </>
            ) : (
              <>
                <Save size={18} />
                {isEditing ? 'Guardar cambios' : 'Crear flujo'}
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  );
}
