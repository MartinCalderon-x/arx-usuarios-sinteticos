import { useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { ArrowLeft, Save, Loader2 } from 'lucide-react';
import { arquetiposApi, type ArquetipoCreate, type ArquetipoTemplate } from '../lib/api';

export function ArquetipoForm() {
  const { id } = useParams();
  const navigate = useNavigate();
  const isEditing = Boolean(id);

  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [templates, setTemplates] = useState<ArquetipoTemplate[]>([]);
  const [form, setForm] = useState<ArquetipoCreate>({
    nombre: '',
    descripcion: '',
    edad: undefined,
    genero: '',
    ocupacion: '',
    contexto: '',
    comportamiento: '',
    frustraciones: [],
    objetivos: [],
  });

  useEffect(() => {
    loadTemplates();
    if (id) loadArquetipo(id);
  }, [id]);

  async function loadTemplates() {
    try {
      const { templates } = await arquetiposApi.templates();
      setTemplates(templates);
    } catch (error) {
      console.error('Error loading templates:', error);
    }
  }

  async function loadArquetipo(id: string) {
    setLoading(true);
    try {
      const arquetipo = await arquetiposApi.get(id);
      setForm({
        nombre: arquetipo.nombre,
        descripcion: arquetipo.descripcion || '',
        edad: arquetipo.edad,
        genero: arquetipo.genero || '',
        ocupacion: arquetipo.ocupacion || '',
        contexto: arquetipo.contexto || '',
        comportamiento: arquetipo.comportamiento || '',
        frustraciones: arquetipo.frustraciones || [],
        objetivos: arquetipo.objetivos || [],
      });
    } catch (error) {
      console.error('Error loading arquetipo:', error);
    } finally {
      setLoading(false);
    }
  }

  function applyTemplate(template: ArquetipoTemplate) {
    setForm(prev => ({
      ...prev,
      nombre: template.nombre,
      descripcion: template.descripcion,
      edad: template.edad,
      ocupacion: template.ocupacion || '',
    }));
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setSaving(true);

    try {
      if (isEditing && id) {
        await arquetiposApi.update(id, form);
      } else {
        await arquetiposApi.create(form);
      }
      navigate('/arquetipos');
    } catch (error) {
      console.error('Error saving arquetipo:', error);
      alert('Error al guardar el arquetipo');
    } finally {
      setSaving(false);
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="w-8 h-8 animate-spin text-primary" />
      </div>
    );
  }

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center gap-4">
        <button
          onClick={() => navigate('/arquetipos')}
          className="p-2 rounded-lg hover:bg-bg-secondary text-text-secondary transition-colors"
        >
          <ArrowLeft size={20} />
        </button>
        <div>
          <h1 className="text-2xl font-bold text-text-primary">
            {isEditing ? 'Editar arquetipo' : 'Nuevo arquetipo'}
          </h1>
          <p className="text-text-secondary mt-1">
            {isEditing ? 'Modifica los datos del arquetipo' : 'Crea un nuevo usuario sintetico'}
          </p>
        </div>
      </div>

      {/* Templates */}
      {!isEditing && templates.length > 0 && (
        <div className="bg-bg-secondary rounded-xl p-4 border border-border">
          <h3 className="text-sm font-medium text-text-secondary mb-3">Usar template</h3>
          <div className="flex flex-wrap gap-2">
            {templates.map(template => (
              <button
                key={template.id}
                onClick={() => applyTemplate(template)}
                className="px-3 py-1.5 text-sm bg-bg-tertiary hover:bg-primary/10 hover:text-primary rounded-lg transition-colors"
              >
                {template.nombre}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Form */}
      <form onSubmit={handleSubmit} className="bg-bg-secondary rounded-xl p-6 border border-border space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="md:col-span-2">
            <label className="block text-sm font-medium text-text-secondary mb-1.5">
              Nombre *
            </label>
            <input
              type="text"
              value={form.nombre}
              onChange={(e) => setForm({ ...form, nombre: e.target.value })}
              className="w-full px-4 py-2.5 rounded-lg border border-border bg-bg-primary text-text-primary focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary"
              placeholder="Ej: Consumidor Digital"
              required
            />
          </div>

          <div className="md:col-span-2">
            <label className="block text-sm font-medium text-text-secondary mb-1.5">
              Descripcion *
            </label>
            <textarea
              value={form.descripcion}
              onChange={(e) => setForm({ ...form, descripcion: e.target.value })}
              className="w-full px-4 py-2.5 rounded-lg border border-border bg-bg-primary text-text-primary focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary resize-none"
              rows={3}
              placeholder="Describe las caracteristicas principales del arquetipo"
              required
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-text-secondary mb-1.5">
              Edad
            </label>
            <input
              type="number"
              value={form.edad || ''}
              onChange={(e) => setForm({ ...form, edad: e.target.value ? parseInt(e.target.value) : undefined })}
              className="w-full px-4 py-2.5 rounded-lg border border-border bg-bg-primary text-text-primary focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary"
              placeholder="Ej: 35"
              min={1}
              max={120}
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-text-secondary mb-1.5">
              Genero
            </label>
            <select
              value={form.genero}
              onChange={(e) => setForm({ ...form, genero: e.target.value })}
              className="w-full px-4 py-2.5 rounded-lg border border-border bg-bg-primary text-text-primary focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary"
            >
              <option value="">Seleccionar</option>
              <option value="masculino">Masculino</option>
              <option value="femenino">Femenino</option>
              <option value="otro">Otro</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-text-secondary mb-1.5">
              Ocupacion
            </label>
            <input
              type="text"
              value={form.ocupacion}
              onChange={(e) => setForm({ ...form, ocupacion: e.target.value })}
              className="w-full px-4 py-2.5 rounded-lg border border-border bg-bg-primary text-text-primary focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary"
              placeholder="Ej: Ingeniero de software"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-text-secondary mb-1.5">
              Contexto
            </label>
            <input
              type="text"
              value={form.contexto}
              onChange={(e) => setForm({ ...form, contexto: e.target.value })}
              className="w-full px-4 py-2.5 rounded-lg border border-border bg-bg-primary text-text-primary focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary"
              placeholder="Ej: Usuario de apps financieras"
            />
          </div>

          <div className="md:col-span-2">
            <label className="block text-sm font-medium text-text-secondary mb-1.5">
              Comportamiento
            </label>
            <textarea
              value={form.comportamiento}
              onChange={(e) => setForm({ ...form, comportamiento: e.target.value })}
              className="w-full px-4 py-2.5 rounded-lg border border-border bg-bg-primary text-text-primary focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary resize-none"
              rows={2}
              placeholder="Describe como se comporta este usuario"
            />
          </div>
        </div>

        <div className="flex justify-end gap-3 pt-4 border-t border-border">
          <button
            type="button"
            onClick={() => navigate('/arquetipos')}
            className="px-4 py-2 text-text-secondary hover:text-text-primary hover:bg-bg-tertiary rounded-lg transition-colors"
          >
            Cancelar
          </button>
          <button
            type="submit"
            disabled={saving}
            className="flex items-center gap-2 px-4 py-2 bg-primary hover:bg-primary-dark text-white rounded-lg transition-colors disabled:opacity-50"
          >
            {saving ? (
              <Loader2 size={18} className="animate-spin" />
            ) : (
              <Save size={18} />
            )}
            {isEditing ? 'Guardar cambios' : 'Crear arquetipo'}
          </button>
        </div>
      </form>
    </div>
  );
}
