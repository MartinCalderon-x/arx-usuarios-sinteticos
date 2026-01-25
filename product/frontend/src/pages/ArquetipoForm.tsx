import { useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { ArrowLeft, Save, Loader2, Plus, X, FileText, Users, PenLine } from 'lucide-react';
import { arquetiposApi, type ArquetipoCreate, type ArquetipoTemplate, type ArquetipoExtraction } from '../lib/api';
import { ArquetipoFromData } from '../components/ArquetipoFromData';

type CreationMode = 'manual' | 'template' | 'data';

const NIVELES_DIGITAL = [
  { value: 'bajo', label: 'Bajo' },
  { value: 'medio', label: 'Medio' },
  { value: 'alto', label: 'Alto' },
];

const INDUSTRIAS = [
  { value: 'tech', label: 'Tecnología' },
  { value: 'salud', label: 'Salud' },
  { value: 'retail', label: 'Retail' },
  { value: 'finanzas', label: 'Finanzas' },
  { value: 'manufactura', label: 'Manufactura' },
  { value: 'educacion', label: 'Educación' },
  { value: 'servicios', label: 'Servicios' },
  { value: 'otro', label: 'Otro' },
];

export function ArquetipoForm() {
  const { id } = useParams();
  const navigate = useNavigate();
  const isEditing = Boolean(id);

  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [templates, setTemplates] = useState<ArquetipoTemplate[]>([]);
  const [selectedCategoria, setSelectedCategoria] = useState<string>('');
  const [creationMode, setCreationMode] = useState<CreationMode>('manual');
  const [extractedCitas, setExtractedCitas] = useState<string[]>([]);
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
    nivel_digital: '',
    industria: '',
  });

  // Para editar listas
  const [newFrustracion, setNewFrustracion] = useState('');
  const [newObjetivo, setNewObjetivo] = useState('');

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
        nivel_digital: arquetipo.nivel_digital || '',
        industria: arquetipo.industria || '',
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
      nivel_digital: template.nivel_digital || '',
      industria: template.industria || '',
      comportamiento: template.comportamiento || '',
      frustraciones: template.frustraciones || [],
      objetivos: template.objetivos || [],
    }));
    setExtractedCitas([]);
  }

  function handleDataExtracted(data: ArquetipoExtraction['extraccion'], citas: string[]) {
    setForm(prev => ({
      ...prev,
      nombre: data.nombre_sugerido || prev.nombre,
      descripcion: data.descripcion || prev.descripcion,
      edad: data.edad_estimada || prev.edad,
      genero: data.genero || prev.genero,
      ocupacion: data.ocupacion || prev.ocupacion,
      contexto: data.contexto || prev.contexto,
      comportamiento: data.comportamiento || prev.comportamiento,
      frustraciones: data.frustraciones.length > 0 ? data.frustraciones : prev.frustraciones,
      objetivos: data.objetivos.length > 0 ? data.objetivos : prev.objetivos,
      nivel_digital: data.nivel_digital || prev.nivel_digital,
      industria: data.industria || prev.industria,
    }));
    setExtractedCitas(citas);
  }

  function addFrustracion() {
    if (!newFrustracion.trim()) return;
    setForm(prev => ({
      ...prev,
      frustraciones: [...(prev.frustraciones || []), newFrustracion.trim()],
    }));
    setNewFrustracion('');
  }

  function removeFrustracion(index: number) {
    setForm(prev => ({
      ...prev,
      frustraciones: prev.frustraciones?.filter((_, i) => i !== index) || [],
    }));
  }

  function addObjetivo() {
    if (!newObjetivo.trim()) return;
    setForm(prev => ({
      ...prev,
      objetivos: [...(prev.objetivos || []), newObjetivo.trim()],
    }));
    setNewObjetivo('');
  }

  function removeObjetivo(index: number) {
    setForm(prev => ({
      ...prev,
      objetivos: prev.objetivos?.filter((_, i) => i !== index) || [],
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

  // Obtener categorías únicas
  const categorias = [...new Set(templates.map(t => t.categoria))];
  const templatesFiltrados = selectedCategoria
    ? templates.filter(t => t.categoria === selectedCategoria)
    : templates;

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

      {/* Creation Mode Selector */}
      {!isEditing && (
        <div className="bg-bg-secondary rounded-xl p-4 border border-border space-y-4">
          {/* Mode tabs */}
          <div className="flex gap-2">
            <button
              onClick={() => setCreationMode('manual')}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                creationMode === 'manual'
                  ? 'bg-primary text-white'
                  : 'bg-bg-tertiary text-text-secondary hover:text-text-primary'
              }`}
            >
              <PenLine size={16} />
              Manual
            </button>
            <button
              onClick={() => setCreationMode('template')}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                creationMode === 'template'
                  ? 'bg-primary text-white'
                  : 'bg-bg-tertiary text-text-secondary hover:text-text-primary'
              }`}
            >
              <Users size={16} />
              Desde plantilla
            </button>
            <button
              onClick={() => setCreationMode('data')}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                creationMode === 'data'
                  ? 'bg-primary text-white'
                  : 'bg-bg-tertiary text-text-secondary hover:text-text-primary'
              }`}
            >
              <FileText size={16} />
              Desde datos
            </button>
          </div>

          {/* Template selector */}
          {creationMode === 'template' && templates.length > 0 && (
            <div className="space-y-3">
              <p className="text-sm text-text-tertiary">
                Selecciona un arquetipo predefinido como base
              </p>
              {/* Filtro por categoría */}
              <div className="flex flex-wrap gap-2">
                <button
                  onClick={() => setSelectedCategoria('')}
                  className={`px-3 py-1 text-xs rounded-full transition-colors ${
                    !selectedCategoria ? 'bg-primary/20 text-primary' : 'bg-bg-tertiary text-text-secondary hover:text-text-primary'
                  }`}
                >
                  Todos
                </button>
                {categorias.map(cat => (
                  <button
                    key={cat}
                    onClick={() => setSelectedCategoria(cat)}
                    className={`px-3 py-1 text-xs rounded-full capitalize transition-colors ${
                      selectedCategoria === cat ? 'bg-primary/20 text-primary' : 'bg-bg-tertiary text-text-secondary hover:text-text-primary'
                    }`}
                  >
                    {cat}
                  </button>
                ))}
              </div>
              {/* Templates */}
              <div className="flex flex-wrap gap-2">
                {templatesFiltrados.map(template => (
                  <button
                    key={template.id}
                    onClick={() => applyTemplate(template)}
                    className="px-3 py-1.5 text-sm bg-bg-tertiary hover:bg-primary/10 hover:text-primary rounded-lg transition-colors text-left"
                    title={template.descripcion}
                  >
                    {template.nombre}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Data extraction */}
          {creationMode === 'data' && (
            <div className="space-y-3">
              <p className="text-sm text-text-tertiary">
                Sube documentos (transcripciones, entrevistas, encuestas) y la IA extraera las caracteristicas del arquetipo
              </p>
              <ArquetipoFromData onExtracted={handleDataExtracted} />
            </div>
          )}

          {/* Manual mode hint */}
          {creationMode === 'manual' && (
            <p className="text-sm text-text-tertiary">
              Completa el formulario manualmente con los datos del arquetipo
            </p>
          )}
        </div>
      )}

      {/* Form */}
      <form onSubmit={handleSubmit} className="bg-bg-secondary rounded-xl p-6 border border-border space-y-6">
        {/* Información básica */}
        <div>
          <h3 className="text-lg font-medium text-text-primary mb-4">Información básica</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="md:col-span-2">
              <label className="block text-sm font-medium text-text-secondary mb-1.5">
                Nombre *
              </label>
              <input
                type="text"
                value={form.nombre}
                onChange={(e) => setForm({ ...form, nombre: e.target.value })}
                className="w-full px-4 py-2.5 rounded-lg border border-border bg-bg-primary text-text-primary focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary"
                placeholder="Ej: Early Adopter Tech"
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
          </div>
        </div>

        {/* Perfil */}
        <div>
          <h3 className="text-lg font-medium text-text-primary mb-4">Perfil</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-text-secondary mb-1.5">
                Nivel Digital
              </label>
              <select
                value={form.nivel_digital}
                onChange={(e) => setForm({ ...form, nivel_digital: e.target.value })}
                className="w-full px-4 py-2.5 rounded-lg border border-border bg-bg-primary text-text-primary focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary"
              >
                <option value="">Seleccionar</option>
                {NIVELES_DIGITAL.map(nivel => (
                  <option key={nivel.value} value={nivel.value}>{nivel.label}</option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-text-secondary mb-1.5">
                Industria
              </label>
              <select
                value={form.industria}
                onChange={(e) => setForm({ ...form, industria: e.target.value })}
                className="w-full px-4 py-2.5 rounded-lg border border-border bg-bg-primary text-text-primary focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary"
              >
                <option value="">Seleccionar</option>
                {INDUSTRIAS.map(ind => (
                  <option key={ind.value} value={ind.value}>{ind.label}</option>
                ))}
              </select>
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
        </div>

        {/* Frustraciones */}
        <div>
          <h3 className="text-lg font-medium text-text-primary mb-4">Frustraciones</h3>
          <div className="space-y-3">
            {form.frustraciones && form.frustraciones.length > 0 && (
              <div className="flex flex-wrap gap-2">
                {form.frustraciones.map((f, i) => (
                  <span
                    key={i}
                    className="inline-flex items-center gap-1 px-3 py-1.5 bg-error/10 text-error rounded-lg text-sm"
                  >
                    {f}
                    <button
                      type="button"
                      onClick={() => removeFrustracion(i)}
                      className="hover:bg-error/20 rounded p-0.5"
                    >
                      <X size={14} />
                    </button>
                  </span>
                ))}
              </div>
            )}
            <div className="flex gap-2">
              <input
                type="text"
                value={newFrustracion}
                onChange={(e) => setNewFrustracion(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && (e.preventDefault(), addFrustracion())}
                className="flex-1 px-4 py-2 rounded-lg border border-border bg-bg-primary text-text-primary focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary"
                placeholder="Agregar frustracion..."
              />
              <button
                type="button"
                onClick={addFrustracion}
                className="px-3 py-2 bg-bg-tertiary hover:bg-error/10 hover:text-error rounded-lg transition-colors"
              >
                <Plus size={18} />
              </button>
            </div>
          </div>
        </div>

        {/* Objetivos */}
        <div>
          <h3 className="text-lg font-medium text-text-primary mb-4">Objetivos</h3>
          <div className="space-y-3">
            {form.objetivos && form.objetivos.length > 0 && (
              <div className="flex flex-wrap gap-2">
                {form.objetivos.map((o, i) => (
                  <span
                    key={i}
                    className="inline-flex items-center gap-1 px-3 py-1.5 bg-success/10 text-success rounded-lg text-sm"
                  >
                    {o}
                    <button
                      type="button"
                      onClick={() => removeObjetivo(i)}
                      className="hover:bg-success/20 rounded p-0.5"
                    >
                      <X size={14} />
                    </button>
                  </span>
                ))}
              </div>
            )}
            <div className="flex gap-2">
              <input
                type="text"
                value={newObjetivo}
                onChange={(e) => setNewObjetivo(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && (e.preventDefault(), addObjetivo())}
                className="flex-1 px-4 py-2 rounded-lg border border-border bg-bg-primary text-text-primary focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary"
                placeholder="Agregar objetivo..."
              />
              <button
                type="button"
                onClick={addObjetivo}
                className="px-3 py-2 bg-bg-tertiary hover:bg-success/10 hover:text-success rounded-lg transition-colors"
              >
                <Plus size={18} />
              </button>
            </div>
          </div>
        </div>

        {/* Extracted quotes reference */}
        {extractedCitas.length > 0 && (
          <div className="border-t border-border pt-4">
            <h3 className="text-lg font-medium text-text-primary mb-3">Citas de referencia</h3>
            <p className="text-sm text-text-tertiary mb-3">
              Citas extraidas de los documentos (solo referencia, no se guardan con el arquetipo)
            </p>
            <div className="space-y-2">
              {extractedCitas.map((cita, i) => (
                <blockquote
                  key={i}
                  className="pl-3 border-l-2 border-primary/30 text-sm text-text-secondary italic"
                >
                  "{cita}"
                </blockquote>
              ))}
            </div>
          </div>
        )}

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
