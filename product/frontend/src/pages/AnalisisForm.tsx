import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, Upload, Link as LinkIcon, Loader2, Eye } from 'lucide-react';
import { analisisApi } from '../lib/api';

type AnalysisType = 'url' | 'file';

export function AnalisisForm() {
  const navigate = useNavigate();
  const [type, setType] = useState<AnalysisType>('url');
  const [url, setUrl] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
    }
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);

    try {
      let result;
      if (type === 'url' && url) {
        result = await analisisApi.analyzeUrl(url);
      } else if (type === 'file' && file) {
        result = await analisisApi.analyzeImage(file);
      } else {
        throw new Error('Por favor ingresa una URL o selecciona un archivo');
      }
      navigate(`/analisis/${result.id}`);
    } catch (error) {
      console.error('Error analyzing:', error);
      alert(error instanceof Error ? error.message : 'Error al analizar');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="max-w-2xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center gap-4">
        <button
          onClick={() => navigate('/analisis')}
          className="p-2 rounded-lg hover:bg-bg-secondary text-text-secondary transition-colors"
        >
          <ArrowLeft size={20} />
        </button>
        <div>
          <h1 className="text-2xl font-bold text-text-primary">Nuevo analisis</h1>
          <p className="text-text-secondary mt-1">Analiza un diseno con Vision AI</p>
        </div>
      </div>

      {/* Type selector */}
      <div className="flex gap-2 p-1 bg-bg-secondary rounded-lg">
        <button
          onClick={() => setType('url')}
          className={`flex-1 flex items-center justify-center gap-2 px-4 py-2 rounded-md transition-colors ${
            type === 'url' ? 'bg-bg-primary shadow text-primary' : 'text-text-secondary hover:text-text-primary'
          }`}
        >
          <LinkIcon size={18} />
          URL
        </button>
        <button
          onClick={() => setType('file')}
          className={`flex-1 flex items-center justify-center gap-2 px-4 py-2 rounded-md transition-colors ${
            type === 'file' ? 'bg-bg-primary shadow text-primary' : 'text-text-secondary hover:text-text-primary'
          }`}
        >
          <Upload size={18} />
          Subir imagen
        </button>
      </div>

      {/* Form */}
      <form onSubmit={handleSubmit} className="bg-bg-secondary rounded-xl p-6 border border-border space-y-6">
        {type === 'url' ? (
          <div>
            <label className="block text-sm font-medium text-text-secondary mb-1.5">
              URL de la imagen
            </label>
            <input
              type="url"
              value={url}
              onChange={(e) => {
                setUrl(e.target.value);
                setPreview(e.target.value || null);
              }}
              className="w-full px-4 py-2.5 rounded-lg border border-border bg-bg-primary text-text-primary focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary"
              placeholder="https://ejemplo.com/imagen.png"
            />
          </div>
        ) : (
          <div>
            <label className="block text-sm font-medium text-text-secondary mb-1.5">
              Seleccionar imagen
            </label>
            <div className="border-2 border-dashed border-border rounded-lg p-8 text-center hover:border-primary/50 transition-colors">
              <input
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                className="hidden"
                id="file-input"
              />
              <label htmlFor="file-input" className="cursor-pointer">
                <Upload size={32} className="mx-auto text-text-muted mb-3" />
                <p className="text-text-secondary mb-1">
                  {file ? file.name : 'Haz clic para seleccionar'}
                </p>
                <p className="text-sm text-text-muted">PNG, JPG hasta 10MB</p>
              </label>
            </div>
          </div>
        )}

        {/* Preview */}
        {preview && (
          <div className="rounded-lg overflow-hidden border border-border">
            <img
              src={preview}
              alt="Preview"
              className="w-full max-h-64 object-contain bg-bg-tertiary"
              onError={() => setPreview(null)}
            />
          </div>
        )}

        <div className="flex justify-end gap-3 pt-4 border-t border-border">
          <button
            type="button"
            onClick={() => navigate('/analisis')}
            className="px-4 py-2 text-text-secondary hover:text-text-primary hover:bg-bg-tertiary rounded-lg transition-colors"
          >
            Cancelar
          </button>
          <button
            type="submit"
            disabled={loading || (!url && !file)}
            className="flex items-center gap-2 px-4 py-2 bg-secondary hover:bg-secondary/90 text-white rounded-lg transition-colors disabled:opacity-50"
          >
            {loading ? (
              <Loader2 size={18} className="animate-spin" />
            ) : (
              <Eye size={18} />
            )}
            Analizar
          </button>
        </div>
      </form>
    </div>
  );
}
