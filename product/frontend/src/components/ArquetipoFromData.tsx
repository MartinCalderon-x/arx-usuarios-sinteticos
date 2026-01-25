import { useState, useCallback } from 'react';
import { Upload, X, FileText, Loader2, AlertCircle, CheckCircle, Quote } from 'lucide-react';
import { arquetiposApi, type ArquetipoExtraction } from '../lib/api';

interface ArquetipoFromDataProps {
  onExtracted: (data: ArquetipoExtraction['extraccion'], citas: string[]) => void;
}

const ACCEPTED_TYPES = [
  'application/pdf',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  'text/plain',
  'text/csv',
  'application/csv',
];

const ACCEPTED_EXTENSIONS = ['.pdf', '.docx', '.txt', '.csv'];

const MAX_FILES = 5;
const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB

export function ArquetipoFromData({ onExtracted }: ArquetipoFromDataProps) {
  const [files, setFiles] = useState<File[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ArquetipoExtraction | null>(null);

  const validateFile = useCallback((file: File): string | null => {
    if (!ACCEPTED_TYPES.includes(file.type)) {
      const ext = file.name.split('.').pop()?.toLowerCase();
      if (!ext || !ACCEPTED_EXTENSIONS.includes(`.${ext}`)) {
        return `${file.name}: Tipo de archivo no soportado`;
      }
    }
    if (file.size > MAX_FILE_SIZE) {
      return `${file.name}: Excede 10MB`;
    }
    return null;
  }, []);

  const addFiles = useCallback((newFiles: FileList | File[]) => {
    const fileArray = Array.from(newFiles);
    const errors: string[] = [];

    // Validate each file
    const validFiles = fileArray.filter(file => {
      const error = validateFile(file);
      if (error) {
        errors.push(error);
        return false;
      }
      return true;
    });

    // Check total count
    const totalCount = files.length + validFiles.length;
    if (totalCount > MAX_FILES) {
      errors.push(`Maximo ${MAX_FILES} archivos permitidos`);
      validFiles.splice(MAX_FILES - files.length);
    }

    // Check for duplicates
    const uniqueFiles = validFiles.filter(
      newFile => !files.some(existingFile => existingFile.name === newFile.name)
    );

    if (errors.length > 0) {
      setError(errors.join('. '));
    } else {
      setError(null);
    }

    setFiles(prev => [...prev, ...uniqueFiles]);
    setResult(null);
  }, [files, validateFile]);

  const removeFile = useCallback((index: number) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
    setResult(null);
    setError(null);
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files) {
      addFiles(e.dataTransfer.files);
    }
  }, [addFiles]);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      addFiles(e.target.files);
    }
    e.target.value = ''; // Reset input
  }, [addFiles]);

  const handleAnalyze = async () => {
    if (files.length === 0) {
      setError('Selecciona al menos un archivo');
      return;
    }

    setIsAnalyzing(true);
    setError(null);

    try {
      const extraction = await arquetiposApi.extractFromData(files);
      setResult(extraction);

      // Notify parent with extracted data
      onExtracted(extraction.extraccion, extraction.citas_relevantes);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Error al analizar documentos');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-success';
    if (confidence >= 0.6) return 'text-warning';
    return 'text-error';
  };

  const getConfidenceLabel = (confidence: number) => {
    if (confidence >= 0.8) return 'Alta';
    if (confidence >= 0.6) return 'Media';
    return 'Baja';
  };

  return (
    <div className="space-y-4">
      {/* Dropzone */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`
          relative border-2 border-dashed rounded-xl p-8 text-center transition-colors
          ${isDragging
            ? 'border-primary bg-primary/5'
            : 'border-border hover:border-primary/50 hover:bg-bg-tertiary/50'
          }
        `}
      >
        <input
          type="file"
          multiple
          accept={ACCEPTED_EXTENSIONS.join(',')}
          onChange={handleFileInput}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
        />
        <Upload className={`w-10 h-10 mx-auto mb-3 ${isDragging ? 'text-primary' : 'text-text-tertiary'}`} />
        <p className="text-text-primary font-medium">
          Arrastra archivos aqui o haz clic para seleccionar
        </p>
        <p className="text-sm text-text-tertiary mt-1">
          PDF, DOCX, TXT, CSV (max {MAX_FILES} archivos, 10MB c/u)
        </p>
      </div>

      {/* File list */}
      {files.length > 0 && (
        <div className="space-y-2">
          <p className="text-sm text-text-secondary">
            Archivos ({files.length}/{MAX_FILES}):
          </p>
          {files.map((file, index) => (
            <div
              key={`${file.name}-${index}`}
              className="flex items-center justify-between px-3 py-2 bg-bg-tertiary rounded-lg"
            >
              <div className="flex items-center gap-2">
                <FileText size={16} className="text-text-tertiary" />
                <span className="text-sm text-text-primary truncate max-w-[250px]">
                  {file.name}
                </span>
                <span className="text-xs text-text-tertiary">
                  ({(file.size / 1024).toFixed(0)} KB)
                </span>
              </div>
              <button
                onClick={() => removeFile(index)}
                className="p-1 hover:bg-error/10 hover:text-error rounded transition-colors"
              >
                <X size={16} />
              </button>
            </div>
          ))}
        </div>
      )}

      {/* Error message */}
      {error && (
        <div className="flex items-start gap-2 px-3 py-2 bg-error/10 text-error rounded-lg text-sm">
          <AlertCircle size={16} className="shrink-0 mt-0.5" />
          <span>{error}</span>
        </div>
      )}

      {/* Analyze button */}
      <button
        onClick={handleAnalyze}
        disabled={files.length === 0 || isAnalyzing}
        className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-primary hover:bg-primary-dark text-white rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {isAnalyzing ? (
          <>
            <Loader2 size={18} className="animate-spin" />
            Analizando documentos...
          </>
        ) : (
          <>
            <Upload size={18} />
            Analizar con IA
          </>
        )}
      </button>

      {/* Results */}
      {result && (
        <div className="bg-bg-tertiary rounded-xl p-4 space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <CheckCircle size={18} className="text-success" />
              <span className="font-medium text-text-primary">Analisis completado</span>
            </div>
            <div className={`flex items-center gap-1 text-sm ${getConfidenceColor(result.confianza)}`}>
              <span>Confianza: {getConfidenceLabel(result.confianza)}</span>
              <span className="font-mono">({Math.round(result.confianza * 100)}%)</span>
            </div>
          </div>

          <div className="text-sm text-text-secondary">
            {result.archivos_procesados} archivo(s) procesado(s)
            {result.archivos_fallidos > 0 && (
              <span className="text-warning"> ({result.archivos_fallidos} fallido(s))</span>
            )}
          </div>

          {/* Extracted quotes */}
          {result.citas_relevantes.length > 0 && (
            <div className="border-t border-border pt-4">
              <div className="flex items-center gap-2 mb-2">
                <Quote size={14} className="text-text-tertiary" />
                <span className="text-sm font-medium text-text-secondary">Citas relevantes</span>
              </div>
              <div className="space-y-2">
                {result.citas_relevantes.map((cita, i) => (
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

          {/* Warnings for file errors */}
          {result.errores_archivos && result.errores_archivos.length > 0 && (
            <div className="text-sm text-warning">
              <p className="font-medium">Advertencias:</p>
              <ul className="list-disc list-inside">
                {result.errores_archivos.map((err, i) => (
                  <li key={i}>{err}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
