-- ============================================
-- Usuarios Sinteticos - Flujos Multi-Pantalla
-- Migration for user journey/flow tracking
-- Issue #15
-- ============================================

-- ============================================
-- Table: us_flujos
-- Main flows/journeys container
-- ============================================

CREATE TABLE IF NOT EXISTS us_flujos (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    nombre VARCHAR(255) NOT NULL,
    descripcion TEXT,
    url_inicial TEXT,
    estado VARCHAR(50) DEFAULT 'activo',
    configuracion JSONB DEFAULT '{}',
    total_pantallas INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for us_flujos
CREATE INDEX IF NOT EXISTS idx_us_flujos_user ON us_flujos(user_id);
CREATE INDEX IF NOT EXISTS idx_us_flujos_estado ON us_flujos(estado);
CREATE INDEX IF NOT EXISTS idx_us_flujos_created ON us_flujos(created_at DESC);

-- ============================================
-- Table: us_pantallas
-- Individual screens within a flow
-- ============================================

CREATE TABLE IF NOT EXISTS us_pantallas (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    flujo_id UUID REFERENCES us_flujos(id) ON DELETE CASCADE,
    user_id UUID REFERENCES auth.users(id),
    orden INTEGER NOT NULL,

    -- Screen source
    origen VARCHAR(50) NOT NULL,
    url TEXT,
    titulo VARCHAR(500),

    -- Storage paths
    screenshot_path TEXT,
    heatmap_path TEXT,
    overlay_path TEXT,

    -- Analysis results
    clarity_score DECIMAL(5,2),
    areas_interes JSONB DEFAULT '[]',
    insights JSONB DEFAULT '[]',
    modelo_usado VARCHAR(100),

    -- Interactive elements detected
    elementos_clickeables JSONB DEFAULT '[]',

    -- Metadata
    viewport JSONB DEFAULT '{"width": 1280, "height": 800}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for us_pantallas
CREATE INDEX IF NOT EXISTS idx_us_pantallas_flujo ON us_pantallas(flujo_id);
CREATE INDEX IF NOT EXISTS idx_us_pantallas_user ON us_pantallas(user_id);
CREATE INDEX IF NOT EXISTS idx_us_pantallas_orden ON us_pantallas(flujo_id, orden);
CREATE INDEX IF NOT EXISTS idx_us_pantallas_origen ON us_pantallas(origen);

-- ============================================
-- Table: us_transiciones
-- Transitions between screens
-- ============================================

CREATE TABLE IF NOT EXISTS us_transiciones (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    flujo_id UUID REFERENCES us_flujos(id) ON DELETE CASCADE,
    pantalla_origen_id UUID REFERENCES us_pantallas(id) ON DELETE CASCADE,
    pantalla_destino_id UUID REFERENCES us_pantallas(id) ON DELETE CASCADE,
    elemento_clickeado JSONB,
    tipo VARCHAR(50) DEFAULT 'manual',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for us_transiciones
CREATE INDEX IF NOT EXISTS idx_us_transiciones_flujo ON us_transiciones(flujo_id);
CREATE INDEX IF NOT EXISTS idx_us_transiciones_origen ON us_transiciones(pantalla_origen_id);
CREATE INDEX IF NOT EXISTS idx_us_transiciones_destino ON us_transiciones(pantalla_destino_id);

-- ============================================
-- Enable Row Level Security
-- ============================================

ALTER TABLE us_flujos ENABLE ROW LEVEL SECURITY;
ALTER TABLE us_pantallas ENABLE ROW LEVEL SECURITY;
ALTER TABLE us_transiciones ENABLE ROW LEVEL SECURITY;

-- ============================================
-- RLS Policies - us_flujos
-- ============================================

CREATE POLICY "us_flujos_select_own" ON us_flujos
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "us_flujos_insert_own" ON us_flujos
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "us_flujos_update_own" ON us_flujos
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "us_flujos_delete_own" ON us_flujos
    FOR DELETE USING (auth.uid() = user_id);

CREATE POLICY "us_flujos_service_role" ON us_flujos
    FOR ALL USING (auth.jwt() ->> 'role' = 'service_role');

-- ============================================
-- RLS Policies - us_pantallas
-- ============================================

CREATE POLICY "us_pantallas_select_own" ON us_pantallas
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "us_pantallas_insert_own" ON us_pantallas
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "us_pantallas_update_own" ON us_pantallas
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "us_pantallas_delete_own" ON us_pantallas
    FOR DELETE USING (auth.uid() = user_id);

CREATE POLICY "us_pantallas_service_role" ON us_pantallas
    FOR ALL USING (auth.jwt() ->> 'role' = 'service_role');

-- ============================================
-- RLS Policies - us_transiciones (via flujo ownership)
-- ============================================

CREATE POLICY "us_transiciones_select_own" ON us_transiciones
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM us_flujos
            WHERE us_flujos.id = us_transiciones.flujo_id
            AND us_flujos.user_id = auth.uid()
        )
    );

CREATE POLICY "us_transiciones_insert_own" ON us_transiciones
    FOR INSERT WITH CHECK (
        EXISTS (
            SELECT 1 FROM us_flujos
            WHERE us_flujos.id = flujo_id
            AND us_flujos.user_id = auth.uid()
        )
    );

CREATE POLICY "us_transiciones_delete_own" ON us_transiciones
    FOR DELETE USING (
        EXISTS (
            SELECT 1 FROM us_flujos
            WHERE us_flujos.id = us_transiciones.flujo_id
            AND us_flujos.user_id = auth.uid()
        )
    );

CREATE POLICY "us_transiciones_service_role" ON us_transiciones
    FOR ALL USING (auth.jwt() ->> 'role' = 'service_role');

-- ============================================
-- Trigger for updated_at on us_flujos
-- ============================================

CREATE OR REPLACE FUNCTION update_us_flujos_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_us_flujos_updated_at
    BEFORE UPDATE ON us_flujos
    FOR EACH ROW
    EXECUTE FUNCTION update_us_flujos_updated_at();

-- ============================================
-- Trigger to update total_pantallas count
-- ============================================

CREATE OR REPLACE FUNCTION update_flujo_pantallas_count()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE us_flujos
        SET total_pantallas = (
            SELECT COUNT(*) FROM us_pantallas WHERE flujo_id = NEW.flujo_id
        )
        WHERE id = NEW.flujo_id;
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE us_flujos
        SET total_pantallas = (
            SELECT COUNT(*) FROM us_pantallas WHERE flujo_id = OLD.flujo_id
        )
        WHERE id = OLD.flujo_id;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_us_pantallas_count
    AFTER INSERT OR DELETE ON us_pantallas
    FOR EACH ROW
    EXECUTE FUNCTION update_flujo_pantallas_count();

-- Refresh PostgREST schema cache
NOTIFY pgrst, 'reload schema';
