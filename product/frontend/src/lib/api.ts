import { supabase } from './supabase';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

async function getAuthHeaders(): Promise<HeadersInit> {
  const { data: { session } } = await supabase.auth.getSession();
  return {
    'Content-Type': 'application/json',
    ...(session?.access_token && { Authorization: `Bearer ${session.access_token}` }),
  };
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Error desconocido' }));
    throw new Error(error.detail || `Error ${response.status}`);
  }
  return response.json();
}

// Helper to clean empty strings from objects before sending to API
function cleanEmptyValues(obj: object): object {
  const cleaned: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(obj)) {
    if (value === '' || value === undefined || value === null) continue;
    if (Array.isArray(value) && value.length === 0) continue;
    cleaned[key] = value;
  }
  return cleaned;
}

// Arquetipos
export const arquetiposApi = {
  list: async () => {
    const headers = await getAuthHeaders();
    const response = await fetch(`${API_URL}/api/arquetipos/`, { headers });
    return handleResponse<{ arquetipos: Arquetipo[]; total: number }>(response);
  },

  get: async (id: string) => {
    const headers = await getAuthHeaders();
    const response = await fetch(`${API_URL}/api/arquetipos/${id}`, { headers });
    return handleResponse<Arquetipo>(response);
  },

  create: async (data: ArquetipoCreate) => {
    const headers = await getAuthHeaders();
    const response = await fetch(`${API_URL}/api/arquetipos/`, {
      method: 'POST',
      headers,
      body: JSON.stringify(cleanEmptyValues(data)),
    });
    return handleResponse<Arquetipo>(response);
  },

  update: async (id: string, data: ArquetipoCreate) => {
    const headers = await getAuthHeaders();
    const response = await fetch(`${API_URL}/api/arquetipos/${id}`, {
      method: 'PUT',
      headers,
      body: JSON.stringify(cleanEmptyValues(data)),
    });
    return handleResponse<Arquetipo>(response);
  },

  delete: async (id: string) => {
    const headers = await getAuthHeaders();
    const response = await fetch(`${API_URL}/api/arquetipos/${id}`, {
      method: 'DELETE',
      headers,
    });
    return handleResponse<{ message: string }>(response);
  },

  templates: async () => {
    const response = await fetch(`${API_URL}/api/arquetipos/templates/`);
    return handleResponse<{ templates: ArquetipoTemplate[] }>(response);
  },

  extractFromData: async (files: File[]) => {
    const { data: { session } } = await supabase.auth.getSession();
    const formData = new FormData();
    files.forEach(file => formData.append('files', file));

    const response = await fetch(`${API_URL}/api/arquetipos/extraer-desde-datos`, {
      method: 'POST',
      headers: {
        ...(session?.access_token && { Authorization: `Bearer ${session.access_token}` }),
      },
      body: formData,
    });
    return handleResponse<ArquetipoExtraction>(response);
  },
};

// Analisis
export const analisisApi = {
  list: async () => {
    const headers = await getAuthHeaders();
    const response = await fetch(`${API_URL}/api/analisis/`, { headers });
    return handleResponse<{ analisis: Analisis[]; total: number }>(response);
  },

  get: async (id: string) => {
    const headers = await getAuthHeaders();
    const response = await fetch(`${API_URL}/api/analisis/${id}`, { headers });
    return handleResponse<Analisis>(response);
  },

  analyzeUrl: async (url: string, tipoAnalisis: string[] = ['heatmap', 'focus_map', 'aoi']) => {
    const headers = await getAuthHeaders();
    const response = await fetch(`${API_URL}/api/analisis/url`, {
      method: 'POST',
      headers,
      body: JSON.stringify({ url, tipo_analisis: tipoAnalisis }),
    });
    return handleResponse<Analisis>(response);
  },

  analyzeImage: async (file: File) => {
    const { data: { session } } = await supabase.auth.getSession();
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_URL}/api/analisis/imagen`, {
      method: 'POST',
      headers: {
        ...(session?.access_token && { Authorization: `Bearer ${session.access_token}` }),
      },
      body: formData,
    });
    return handleResponse<Analisis>(response);
  },

  compare: async (imagenAUrl: string, imagenBUrl: string) => {
    const headers = await getAuthHeaders();
    const response = await fetch(`${API_URL}/api/analisis/comparar?imagen_a_url=${encodeURIComponent(imagenAUrl)}&imagen_b_url=${encodeURIComponent(imagenBUrl)}`, {
      method: 'POST',
      headers,
    });
    return handleResponse<AnalisisComparison>(response);
  },

  compareModels: async (file: File) => {
    const { data: { session } } = await supabase.auth.getSession();
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_URL}/api/analisis/comparar-modelos`, {
      method: 'POST',
      headers: {
        ...(session?.access_token && { Authorization: `Bearer ${session.access_token}` }),
      },
      body: formData,
    });
    return handleResponse<ModelComparisonResponse>(response);
  },
};

// Interaccion
export const interaccionApi = {
  chat: async (arquetipoId: string, mensaje: string, sessionId?: string, imagenUrl?: string) => {
    const headers = await getAuthHeaders();
    const response = await fetch(`${API_URL}/api/interaccion/chat`, {
      method: 'POST',
      headers,
      body: JSON.stringify({
        arquetipo_id: arquetipoId,
        mensaje,
        session_id: sessionId,
        imagen_url: imagenUrl,
      }),
    });
    return handleResponse<ChatResponse>(response);
  },

  evaluate: async (arquetipoId: string, imagenUrl: string, contexto?: string, preguntas?: string[]) => {
    const headers = await getAuthHeaders();
    const response = await fetch(`${API_URL}/api/interaccion/evaluar`, {
      method: 'POST',
      headers,
      body: JSON.stringify({
        arquetipo_id: arquetipoId,
        imagen_url: imagenUrl,
        contexto,
        preguntas,
      }),
    });
    return handleResponse<EvaluacionResponse>(response);
  },

  getHistory: async (sessionId: string) => {
    const headers = await getAuthHeaders();
    const response = await fetch(`${API_URL}/api/interaccion/historial/${sessionId}`, { headers });
    return handleResponse<{ session_id: string; mensajes: Mensaje[]; total: number }>(response);
  },

  listSessions: async (arquetipoId?: string) => {
    const headers = await getAuthHeaders();
    const url = arquetipoId
      ? `${API_URL}/api/interaccion/sesiones?arquetipo_id=${arquetipoId}`
      : `${API_URL}/api/interaccion/sesiones`;
    const response = await fetch(url, { headers });
    return handleResponse<{ sesiones: Sesion[]; total: number }>(response);
  },
};

// Reportes
export const reportesApi = {
  list: async () => {
    const headers = await getAuthHeaders();
    const response = await fetch(`${API_URL}/api/reportes/`, { headers });
    return handleResponse<{ reportes: Reporte[]; total: number }>(response);
  },

  generate: async (data: ReporteRequest) => {
    const headers = await getAuthHeaders();
    const response = await fetch(`${API_URL}/api/reportes/generar`, {
      method: 'POST',
      headers,
      body: JSON.stringify(data),
    });
    return handleResponse<Reporte>(response);
  },

  download: async (id: string) => {
    const { data: { session } } = await supabase.auth.getSession();
    const response = await fetch(`${API_URL}/api/reportes/${id}/descargar`, {
      headers: {
        ...(session?.access_token && { Authorization: `Bearer ${session.access_token}` }),
      },
    });
    if (!response.ok) {
      throw new Error('Error al descargar reporte');
    }
    return response.blob();
  },

  delete: async (id: string) => {
    const headers = await getAuthHeaders();
    const response = await fetch(`${API_URL}/api/reportes/${id}`, {
      method: 'DELETE',
      headers,
    });
    return handleResponse<{ message: string }>(response);
  },
};

// Types
export interface Arquetipo {
  id: string;
  nombre: string;
  descripcion?: string;
  edad?: number;
  genero?: string;
  ocupacion?: string;
  contexto?: string;
  comportamiento?: string;
  frustraciones?: string[];
  objetivos?: string[];
  nivel_digital?: string;
  industria?: string;
  created_at?: string;
}

export interface ArquetipoCreate {
  nombre: string;
  descripcion: string;
  edad?: number;
  genero?: string;
  ocupacion?: string;
  contexto?: string;
  comportamiento?: string;
  frustraciones?: string[];
  objetivos?: string[];
  nivel_digital?: string;
  industria?: string;
  template_id?: string;
}

export interface ArquetipoTemplate {
  id: string;
  nombre: string;
  categoria: string;
  descripcion: string;
  edad?: number;
  ocupacion?: string;
  nivel_digital?: string;
  industria?: string;
  comportamiento?: string;
  frustraciones?: string[];
  objetivos?: string[];
}

export interface ArquetipoExtraction {
  extraccion: {
    nombre_sugerido: string;
    descripcion: string;
    edad_estimada?: number;
    genero?: string;
    ocupacion?: string;
    contexto?: string;
    comportamiento?: string;
    frustraciones: string[];
    objetivos: string[];
    nivel_digital?: string;
    industria?: string;
  };
  citas_relevantes: string[];
  confianza: number;
  archivos_procesados: number;
  archivos_fallidos: number;
  errores_archivos?: string[];
}

export interface Analisis {
  id: string;
  imagen_url: string;
  heatmap_url?: string;
  focus_map_url?: string;
  clarity_score?: number;
  areas_interes?: AreaInteres[];
  insights?: string[];
  modelo_usado?: string;
  created_at?: string;
}

export interface AreaInteres {
  nombre: string;
  x: number;
  y: number;
  width: number;
  height: number;
  intensidad: number;
  orden_visual: number;
}

export interface AnalisisComparison {
  imagen_a: { url: string; analisis: Analisis };
  imagen_b: { url: string; analisis: Analisis };
  comparacion: {
    ganador: string;
    diferencias_clave: string[];
    recomendaciones: string[];
  };
}

export interface ChatResponse {
  respuesta: string;
  session_id: string;
  fricciones?: string[];
  emociones?: Record<string, unknown>;
}

export interface EvaluacionResponse {
  feedback: string;
  puntuacion?: number;
  fricciones: string[];
  sugerencias: string[];
  emociones: Record<string, unknown>;
}

export interface Mensaje {
  id: string;
  rol: 'usuario' | 'sintetico';
  contenido: string;
  imagen_url?: string;
  fricciones?: string[];
  emociones?: Record<string, unknown>;
  created_at: string;
}

export interface Sesion {
  id: string;
  arquetipo_id: string;
  contexto?: string;
  estado: string;
  created_at: string;
}

export interface Reporte {
  id: string;
  titulo: string;
  formato: string;
  url?: string;
  created_at: string;
}

export interface ReporteRequest {
  titulo: string;
  formato?: 'pdf' | 'pptx';
  arquetipos_ids?: string[];
  analisis_ids?: string[];
  sesiones_ids?: string[];
  incluir_resumen?: boolean;
  incluir_recomendaciones?: boolean;
}

// Model Comparison Types
export interface ModelResult {
  heatmap_base64: string;
  heatmap_overlay_base64?: string;
  regions: AttentionRegion[];
  inference_time_ms: number;
  model_name: string;
}

export interface AttentionRegion {
  x: number;
  y: number;
  width: number;
  height: number;
  intensity: number;
  area: number;
  orden_visual: number;
}

export interface ComparisonMetrics {
  correlation_coefficient: number;
  kl_divergence: number;
  similarity: number;
  nss: number;
  alignment_percentage: number;
  verdict: string;
}

export interface ModelComparisonResponse {
  ml_model?: ModelResult;
  hybrid_model: ModelResult;
  comparison?: ComparisonMetrics;
  ml_service_available: boolean;
  total_time_ms: number;
}
