import { useState, useCallback } from 'react';
import { analisisApi, ModelComparisonResponse } from '../lib/api';

export default function ModelComparison() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ModelComparisonResponse | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleFileSelect = useCallback((selectedFile: File) => {
    if (!selectedFile.type.startsWith('image/')) {
      setError('Por favor selecciona una imagen válida');
      return;
    }
    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setError(null);
    setResult(null);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile) {
      handleFileSelect(droppedFile);
    }
  }, [handleFileSelect]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setIsDragging(false);
  }, []);

  const handleAnalyze = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);

    try {
      const response = await analisisApi.compareModels(file);
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Error al analizar imagen');
    } finally {
      setLoading(false);
    }
  };

  const formatMetric = (value: number, decimals: number = 2) => {
    return value.toFixed(decimals);
  };

  const getMetricColor = (value: number, threshold: number = 0.7) => {
    if (value >= threshold) return 'text-green-400';
    if (value >= threshold * 0.7) return 'text-yellow-400';
    return 'text-red-400';
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-semibold text-text-primary">
          Comparación de Modelos de Atención Visual
        </h1>
        <p className="text-text-secondary mt-1">
          Compara predicciones de DeepGaze (ML) vs Híbrido (Gemini + Gaussian)
        </p>
      </div>

      {/* Upload Area */}
      <div className="bg-bg-secondary rounded-lg p-6 border border-border">
        <div
          className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
            isDragging
              ? 'border-accent-primary bg-accent-primary/10'
              : 'border-border hover:border-accent-primary/50'
          }`}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
        >
          {preview ? (
            <div className="space-y-4">
              <img
                src={preview}
                alt="Preview"
                className="max-h-64 mx-auto rounded-lg"
              />
              <p className="text-text-secondary">{file?.name}</p>
              <button
                onClick={() => {
                  setFile(null);
                  setPreview(null);
                  setResult(null);
                }}
                className="text-sm text-accent-primary hover:underline"
              >
                Cambiar imagen
              </button>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="text-4xl text-text-tertiary">📷</div>
              <p className="text-text-secondary">
                Arrastra una imagen aquí o{' '}
                <label className="text-accent-primary cursor-pointer hover:underline">
                  selecciona un archivo
                  <input
                    type="file"
                    accept="image/*"
                    className="hidden"
                    onChange={(e) => e.target.files?.[0] && handleFileSelect(e.target.files[0])}
                  />
                </label>
              </p>
              <p className="text-sm text-text-tertiary">
                PNG, JPG hasta 10MB
              </p>
            </div>
          )}
        </div>

        {file && (
          <div className="mt-4 flex justify-center">
            <button
              onClick={handleAnalyze}
              disabled={loading}
              className="px-6 py-2 bg-accent-primary text-white rounded-lg hover:bg-accent-primary/90 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
            >
              {loading ? (
                <>
                  <span className="animate-spin">⏳</span>
                  Analizando...
                </>
              ) : (
                <>
                  🔬 Comparar Modelos
                </>
              )}
            </button>
          </div>
        )}

        {error && (
          <div className="mt-4 p-4 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400">
            {error}
          </div>
        )}
      </div>

      {/* Results */}
      {result && (
        <div className="space-y-6">
          {/* Status Banner */}
          <div className={`p-4 rounded-lg ${
            result.ml_service_available
              ? 'bg-green-500/10 border border-green-500/30'
              : 'bg-yellow-500/10 border border-yellow-500/30'
          }`}>
            <div className="flex items-center justify-between">
              <div>
                <span className={result.ml_service_available ? 'text-green-400' : 'text-yellow-400'}>
                  {result.ml_service_available
                    ? '✓ ML Service (DeepGaze) disponible'
                    : '⚠ ML Service no disponible - Solo mostrando modelo híbrido'}
                </span>
              </div>
              <div className="text-text-secondary text-sm">
                Tiempo total: {result.total_time_ms.toFixed(0)}ms
              </div>
            </div>
          </div>

          {/* Heatmaps Comparison */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* DeepGaze Result */}
            <div className="bg-bg-secondary rounded-lg p-4 border border-border">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-medium text-text-primary">
                  🧠 DeepGaze (ML)
                </h3>
                {result.ml_model && (
                  <span className="text-sm text-text-secondary">
                    {result.ml_model.inference_time_ms.toFixed(0)}ms
                  </span>
                )}
              </div>

              {result.ml_model ? (
                <div className="space-y-4">
                  <div className="bg-bg-tertiary rounded-lg overflow-hidden">
                    <img
                      src={`data:image/png;base64,${result.ml_model.heatmap_overlay_base64 || result.ml_model.heatmap_base64}`}
                      alt="DeepGaze Heatmap"
                      className="w-full"
                    />
                  </div>
                  <div className="text-sm text-text-secondary">
                    <span className="font-medium">Ground Truth</span> - Modelo entrenado con datos reales de eye-tracking
                  </div>
                  <div className="text-xs text-text-tertiary">
                    {result.ml_model.regions.length} regiones detectadas
                  </div>
                </div>
              ) : (
                <div className="bg-bg-tertiary rounded-lg p-8 text-center text-text-tertiary">
                  <div className="text-4xl mb-2">🔌</div>
                  <p>ML Service no disponible</p>
                  <p className="text-sm mt-1">Conecta el servicio para comparar</p>
                </div>
              )}
            </div>

            {/* Hybrid Result */}
            <div className="bg-bg-secondary rounded-lg p-4 border border-border">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-medium text-text-primary">
                  ⚡ Híbrido (Gemini + Gaussian)
                </h3>
                <span className="text-sm text-text-secondary">
                  {result.hybrid_model.inference_time_ms.toFixed(0)}ms
                </span>
              </div>

              <div className="space-y-4">
                <div className="bg-bg-tertiary rounded-lg overflow-hidden">
                  <img
                    src={`data:image/png;base64,${result.hybrid_model.heatmap_overlay_base64 || result.hybrid_model.heatmap_base64}`}
                    alt="Hybrid Heatmap"
                    className="w-full"
                  />
                </div>
                <div className="text-sm text-text-secondary">
                  <span className="font-medium">Modelo Alternativo</span> - Análisis semántico + interpolación gaussiana
                </div>
                <div className="text-xs text-text-tertiary">
                  {result.hybrid_model.regions.length} regiones detectadas
                </div>
              </div>
            </div>
          </div>

          {/* Comparison Metrics */}
          {result.comparison && (
            <div className="bg-bg-secondary rounded-lg p-6 border border-border">
              <h3 className="text-lg font-medium text-text-primary mb-4">
                📊 Métricas de Comparación
              </h3>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                {/* Correlation */}
                <div className="bg-bg-tertiary rounded-lg p-4">
                  <div className="text-sm text-text-secondary mb-1">Correlación (CC)</div>
                  <div className={`text-2xl font-bold ${getMetricColor(result.comparison.correlation_coefficient)}`}>
                    {formatMetric(result.comparison.correlation_coefficient)}
                  </div>
                  <div className="text-xs text-text-tertiary">Mayor = mejor</div>
                </div>

                {/* Similarity */}
                <div className="bg-bg-tertiary rounded-lg p-4">
                  <div className="text-sm text-text-secondary mb-1">Similitud</div>
                  <div className={`text-2xl font-bold ${getMetricColor(result.comparison.similarity)}`}>
                    {formatMetric(result.comparison.similarity)}
                  </div>
                  <div className="text-xs text-text-tertiary">Mayor = mejor</div>
                </div>

                {/* KL Divergence */}
                <div className="bg-bg-tertiary rounded-lg p-4">
                  <div className="text-sm text-text-secondary mb-1">KL Divergence</div>
                  <div className={`text-2xl font-bold ${getMetricColor(1 - Math.min(result.comparison.kl_divergence, 1))}`}>
                    {formatMetric(result.comparison.kl_divergence)}
                  </div>
                  <div className="text-xs text-text-tertiary">Menor = mejor</div>
                </div>

                {/* Alignment */}
                <div className="bg-bg-tertiary rounded-lg p-4">
                  <div className="text-sm text-text-secondary mb-1">Alineamiento</div>
                  <div className={`text-2xl font-bold ${getMetricColor(result.comparison.alignment_percentage / 100)}`}>
                    {formatMetric(result.comparison.alignment_percentage, 1)}%
                  </div>
                  <div className="text-xs text-text-tertiary">Mayor = mejor</div>
                </div>
              </div>

              {/* Progress Bar */}
              <div className="mb-4">
                <div className="flex justify-between text-sm mb-2">
                  <span className="text-text-secondary">Alineamiento con Ground Truth</span>
                  <span className={getMetricColor(result.comparison.alignment_percentage / 100)}>
                    {formatMetric(result.comparison.alignment_percentage, 1)}%
                  </span>
                </div>
                <div className="h-3 bg-bg-tertiary rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all ${
                      result.comparison.alignment_percentage >= 70
                        ? 'bg-green-500'
                        : result.comparison.alignment_percentage >= 50
                        ? 'bg-yellow-500'
                        : 'bg-red-500'
                    }`}
                    style={{ width: `${Math.min(result.comparison.alignment_percentage, 100)}%` }}
                  />
                </div>
              </div>

              {/* Verdict */}
              <div className="bg-bg-tertiary rounded-lg p-4">
                <div className="text-sm text-text-secondary mb-1">Veredicto</div>
                <div className="text-text-primary">{result.comparison.verdict}</div>
              </div>
            </div>
          )}

          {/* Regions Table */}
          {result.hybrid_model.regions.length > 0 && (
            <div className="bg-bg-secondary rounded-lg p-6 border border-border">
              <h3 className="text-lg font-medium text-text-primary mb-4">
                🎯 Regiones de Atención Detectadas
              </h3>

              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-text-secondary border-b border-border">
                      <th className="pb-2 pr-4">Orden</th>
                      <th className="pb-2 pr-4">Posición</th>
                      <th className="pb-2 pr-4">Tamaño</th>
                      <th className="pb-2">Intensidad</th>
                    </tr>
                  </thead>
                  <tbody className="text-text-primary">
                    {result.hybrid_model.regions.slice(0, 10).map((region, idx) => (
                      <tr key={idx} className="border-b border-border/50">
                        <td className="py-2 pr-4">
                          <span className="inline-flex items-center justify-center w-6 h-6 rounded-full bg-accent-primary/20 text-accent-primary text-xs">
                            {region.orden_visual}
                          </span>
                        </td>
                        <td className="py-2 pr-4">
                          ({region.x.toFixed(1)}%, {region.y.toFixed(1)}%)
                        </td>
                        <td className="py-2 pr-4">
                          {region.width.toFixed(1)}% × {region.height.toFixed(1)}%
                        </td>
                        <td className="py-2">
                          <div className="flex items-center gap-2">
                            <div className="w-16 h-2 bg-bg-tertiary rounded-full overflow-hidden">
                              <div
                                className="h-full bg-accent-primary rounded-full"
                                style={{ width: `${region.intensity}%` }}
                              />
                            </div>
                            <span>{region.intensity.toFixed(0)}%</span>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Technical Info */}
          <div className="bg-bg-secondary rounded-lg p-6 border border-border">
            <h3 className="text-lg font-medium text-text-primary mb-4">
              ℹ️ Información Técnica
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm">
              <div>
                <h4 className="font-medium text-text-primary mb-2">DeepGaze III</h4>
                <ul className="space-y-1 text-text-secondary">
                  <li>• Backbone: ResNet-50 pre-entrenado</li>
                  <li>• Dataset: MIT1003 (eye-tracking real)</li>
                  <li>• Publicación: Journal of Vision, 2022</li>
                  <li>• Autor: Kümmerer & Bethge</li>
                </ul>
              </div>
              <div>
                <h4 className="font-medium text-text-primary mb-2">Modelo Híbrido</h4>
                <ul className="space-y-1 text-text-secondary">
                  <li>• Análisis semántico: Gemini Vision</li>
                  <li>• Interpolación: Gaussian + Center Bias</li>
                  <li>• Ventaja: ~3x más rápido, sin GPU</li>
                  <li>• Limitación: Aproximación, no eye-tracking</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
