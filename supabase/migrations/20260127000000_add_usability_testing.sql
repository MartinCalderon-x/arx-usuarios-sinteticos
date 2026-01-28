-- ============================================
-- Usuarios Sinteticos - Usability Testing
-- Migration for missions and simulations
-- ============================================

-- ============================================
-- Table: us_misiones
-- Usability testing missions/tasks
-- ============================================

CREATE TABLE IF NOT EXISTS us_misiones (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    flujo_id UUID REFERENCES us_flujos(id) ON DELETE CASCADE,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    nombre VARCHAR(255) NOT NULL,
    instrucciones TEXT NOT NULL,
    pantalla_inicio_id UUID REFERENCES us_pantallas(id) ON DELETE SET NULL,
    pantalla_objetivo_id UUID REFERENCES us_pantallas(id) ON DELETE SET NULL,
    elemento_objetivo JSONB,
    max_pasos INTEGER DEFAULT 10,
    estado VARCHAR(50) DEFAULT 'activa',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for us_misiones
CREATE INDEX IF NOT EXISTS idx_us_misiones_flujo ON us_misiones(flujo_id);
CREATE INDEX IF NOT EXISTS idx_us_misiones_user ON us_misiones(user_id);
CREATE INDEX IF NOT EXISTS idx_us_misiones_estado ON us_misiones(estado);
CREATE INDEX IF NOT EXISTS idx_us_misiones_created ON us_misiones(created_at DESC);

-- ============================================
-- Table: us_simulaciones
-- Simulation results per archetype
-- ============================================

CREATE TABLE IF NOT EXISTS us_simulaciones (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    mision_id UUID REFERENCES us_misiones(id) ON DELETE CASCADE,
    arquetipo_id UUID REFERENCES us_arquetipos(id) ON DELETE CASCADE,
    user_id UUID REFERENCES auth.users(id),
    completada BOOLEAN DEFAULT FALSE,
    exito BOOLEAN DEFAULT FALSE,
    path_tomado JSONB DEFAULT '[]',
    pasos_totales INTEGER DEFAULT 0,
    misclicks INTEGER DEFAULT 0,
    tiempo_estimado_ms INTEGER,
    fricciones JSONB DEFAULT '[]',
    feedback_arquetipo TEXT,
    emociones JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for us_simulaciones
CREATE INDEX IF NOT EXISTS idx_us_simulaciones_mision ON us_simulaciones(mision_id);
CREATE INDEX IF NOT EXISTS idx_us_simulaciones_arquetipo ON us_simulaciones(arquetipo_id);
CREATE INDEX IF NOT EXISTS idx_us_simulaciones_user ON us_simulaciones(user_id);
CREATE INDEX IF NOT EXISTS idx_us_simulaciones_exito ON us_simulaciones(exito);
CREATE INDEX IF NOT EXISTS idx_us_simulaciones_created ON us_simulaciones(created_at DESC);

-- ============================================
-- Enable Row Level Security
-- ============================================

ALTER TABLE us_misiones ENABLE ROW LEVEL SECURITY;
ALTER TABLE us_simulaciones ENABLE ROW LEVEL SECURITY;

-- ============================================
-- RLS Policies - us_misiones
-- ============================================

CREATE POLICY "us_misiones_select_own" ON us_misiones
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "us_misiones_insert_own" ON us_misiones
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "us_misiones_update_own" ON us_misiones
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "us_misiones_delete_own" ON us_misiones
    FOR DELETE USING (auth.uid() = user_id);

CREATE POLICY "us_misiones_service_role" ON us_misiones
    FOR ALL USING (auth.jwt() ->> 'role' = 'service_role');

-- ============================================
-- RLS Policies - us_simulaciones
-- ============================================

CREATE POLICY "us_simulaciones_select_own" ON us_simulaciones
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "us_simulaciones_insert_own" ON us_simulaciones
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "us_simulaciones_update_own" ON us_simulaciones
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "us_simulaciones_delete_own" ON us_simulaciones
    FOR DELETE USING (auth.uid() = user_id);

CREATE POLICY "us_simulaciones_service_role" ON us_simulaciones
    FOR ALL USING (auth.jwt() ->> 'role' = 'service_role');

-- ============================================
-- Trigger for updated_at on us_misiones
-- ============================================

CREATE OR REPLACE FUNCTION update_us_misiones_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_us_misiones_updated_at
    BEFORE UPDATE ON us_misiones
    FOR EACH ROW
    EXECUTE FUNCTION update_us_misiones_updated_at();

-- Refresh PostgREST schema cache
NOTIFY pgrst, 'reload schema';
