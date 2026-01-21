-- ============================================
-- Usuarios Sintéticos - Schema Inicial
-- Prefijo: us_ para identificar tablas
-- ============================================

-- ============================================
-- Tabla: us_arquetipos (Synthetic Users)
-- ============================================
CREATE TABLE IF NOT EXISTS us_arquetipos (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    nombre VARCHAR(255) NOT NULL,
    descripcion TEXT,
    edad INTEGER,
    genero VARCHAR(50),
    ocupacion VARCHAR(255),
    contexto TEXT,
    comportamiento TEXT,
    frustraciones JSONB DEFAULT '[]',
    objetivos JSONB DEFAULT '[]',
    variables_extra JSONB DEFAULT '{}',
    template_id VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Índices
CREATE INDEX IF NOT EXISTS idx_us_arquetipos_template ON us_arquetipos(template_id);
CREATE INDEX IF NOT EXISTS idx_us_arquetipos_created ON us_arquetipos(created_at DESC);

-- ============================================
-- Tabla: us_analisis (Visual Analysis Results)
-- ============================================
CREATE TABLE IF NOT EXISTS us_analisis (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    imagen_url TEXT NOT NULL,
    imagen_storage_path TEXT,
    tipo_analisis JSONB DEFAULT '["heatmap", "focus_map", "aoi"]',
    resultados JSONB DEFAULT '{}',
    heatmap_url TEXT,
    focus_map_url TEXT,
    clarity_score DECIMAL(5,2),
    areas_interes JSONB DEFAULT '[]',
    insights JSONB DEFAULT '[]',
    modelo_usado VARCHAR(100) DEFAULT 'gemini-vision',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Índices
CREATE INDEX IF NOT EXISTS idx_us_analisis_created ON us_analisis(created_at DESC);

-- ============================================
-- Tabla: us_sesiones (Interaction Sessions)
-- ============================================
CREATE TABLE IF NOT EXISTS us_sesiones (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    arquetipo_id UUID REFERENCES us_arquetipos(id) ON DELETE SET NULL,
    contexto TEXT,
    metadata JSONB DEFAULT '{}',
    estado VARCHAR(50) DEFAULT 'activa',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Índices
CREATE INDEX IF NOT EXISTS idx_us_sesiones_arquetipo ON us_sesiones(arquetipo_id);
CREATE INDEX IF NOT EXISTS idx_us_sesiones_estado ON us_sesiones(estado);

-- ============================================
-- Tabla: us_mensajes (Chat Messages)
-- ============================================
CREATE TABLE IF NOT EXISTS us_mensajes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES us_sesiones(id) ON DELETE CASCADE,
    rol VARCHAR(50) NOT NULL, -- 'usuario' o 'sintetico'
    contenido TEXT NOT NULL,
    imagen_url TEXT,
    fricciones JSONB DEFAULT '[]',
    emociones JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Índices
CREATE INDEX IF NOT EXISTS idx_us_mensajes_session ON us_mensajes(session_id);
CREATE INDEX IF NOT EXISTS idx_us_mensajes_created ON us_mensajes(created_at);

-- ============================================
-- Tabla: us_reportes (Generated Reports)
-- ============================================
CREATE TABLE IF NOT EXISTS us_reportes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    titulo VARCHAR(255) NOT NULL,
    formato VARCHAR(10) DEFAULT 'pdf', -- 'pdf' o 'pptx'
    archivo_url TEXT,
    archivo_storage_path TEXT,
    contenido JSONB DEFAULT '{}',
    arquetipos_ids JSONB DEFAULT '[]',
    analisis_ids JSONB DEFAULT '[]',
    sesiones_ids JSONB DEFAULT '[]',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Índices
CREATE INDEX IF NOT EXISTS idx_us_reportes_created ON us_reportes(created_at DESC);

-- ============================================
-- Función: Actualizar updated_at automáticamente
-- ============================================
CREATE OR REPLACE FUNCTION us_update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers para updated_at
DROP TRIGGER IF EXISTS us_arquetipos_updated_at ON us_arquetipos;
CREATE TRIGGER us_arquetipos_updated_at
    BEFORE UPDATE ON us_arquetipos
    FOR EACH ROW EXECUTE FUNCTION us_update_updated_at();

DROP TRIGGER IF EXISTS us_sesiones_updated_at ON us_sesiones;
CREATE TRIGGER us_sesiones_updated_at
    BEFORE UPDATE ON us_sesiones
    FOR EACH ROW EXECUTE FUNCTION us_update_updated_at();

-- ============================================
-- Row Level Security (RLS) - Opcional
-- Descomentar si se necesita auth por usuario
-- ============================================
-- ALTER TABLE us_arquetipos ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE us_analisis ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE us_sesiones ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE us_mensajes ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE us_reportes ENABLE ROW LEVEL SECURITY;
