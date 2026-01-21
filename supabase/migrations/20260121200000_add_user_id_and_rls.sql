-- ============================================
-- Usuarios Sintéticos - Add user_id and RLS
-- Migration to add user ownership and policies
-- ============================================

-- ============================================
-- Add user_id columns to all tables
-- ============================================

-- us_arquetipos
ALTER TABLE us_arquetipos
ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE;

CREATE INDEX IF NOT EXISTS idx_us_arquetipos_user ON us_arquetipos(user_id);

-- us_analisis
ALTER TABLE us_analisis
ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE;

CREATE INDEX IF NOT EXISTS idx_us_analisis_user ON us_analisis(user_id);

-- us_sesiones
ALTER TABLE us_sesiones
ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE;

CREATE INDEX IF NOT EXISTS idx_us_sesiones_user ON us_sesiones(user_id);

-- us_reportes
ALTER TABLE us_reportes
ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE;

CREATE INDEX IF NOT EXISTS idx_us_reportes_user ON us_reportes(user_id);

-- ============================================
-- Enable Row Level Security
-- ============================================

ALTER TABLE us_arquetipos ENABLE ROW LEVEL SECURITY;
ALTER TABLE us_analisis ENABLE ROW LEVEL SECURITY;
ALTER TABLE us_sesiones ENABLE ROW LEVEL SECURITY;
ALTER TABLE us_mensajes ENABLE ROW LEVEL SECURITY;
ALTER TABLE us_reportes ENABLE ROW LEVEL SECURITY;

-- ============================================
-- RLS Policies - us_arquetipos
-- ============================================

-- Users can view their own archetypes
CREATE POLICY "us_arquetipos_select_own" ON us_arquetipos
    FOR SELECT USING (auth.uid() = user_id);

-- Users can insert their own archetypes
CREATE POLICY "us_arquetipos_insert_own" ON us_arquetipos
    FOR INSERT WITH CHECK (auth.uid() = user_id);

-- Users can update their own archetypes
CREATE POLICY "us_arquetipos_update_own" ON us_arquetipos
    FOR UPDATE USING (auth.uid() = user_id);

-- Users can delete their own archetypes
CREATE POLICY "us_arquetipos_delete_own" ON us_arquetipos
    FOR DELETE USING (auth.uid() = user_id);

-- Service role can do anything
CREATE POLICY "us_arquetipos_service_role" ON us_arquetipos
    FOR ALL USING (auth.jwt() ->> 'role' = 'service_role');

-- ============================================
-- RLS Policies - us_analisis
-- ============================================

CREATE POLICY "us_analisis_select_own" ON us_analisis
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "us_analisis_insert_own" ON us_analisis
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "us_analisis_update_own" ON us_analisis
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "us_analisis_delete_own" ON us_analisis
    FOR DELETE USING (auth.uid() = user_id);

CREATE POLICY "us_analisis_service_role" ON us_analisis
    FOR ALL USING (auth.jwt() ->> 'role' = 'service_role');

-- ============================================
-- RLS Policies - us_sesiones
-- ============================================

CREATE POLICY "us_sesiones_select_own" ON us_sesiones
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "us_sesiones_insert_own" ON us_sesiones
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "us_sesiones_update_own" ON us_sesiones
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "us_sesiones_delete_own" ON us_sesiones
    FOR DELETE USING (auth.uid() = user_id);

CREATE POLICY "us_sesiones_service_role" ON us_sesiones
    FOR ALL USING (auth.jwt() ->> 'role' = 'service_role');

-- ============================================
-- RLS Policies - us_mensajes (via session ownership)
-- ============================================

CREATE POLICY "us_mensajes_select_own" ON us_mensajes
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM us_sesiones
            WHERE us_sesiones.id = us_mensajes.session_id
            AND us_sesiones.user_id = auth.uid()
        )
    );

CREATE POLICY "us_mensajes_insert_own" ON us_mensajes
    FOR INSERT WITH CHECK (
        EXISTS (
            SELECT 1 FROM us_sesiones
            WHERE us_sesiones.id = session_id
            AND us_sesiones.user_id = auth.uid()
        )
    );

CREATE POLICY "us_mensajes_service_role" ON us_mensajes
    FOR ALL USING (auth.jwt() ->> 'role' = 'service_role');

-- ============================================
-- RLS Policies - us_reportes
-- ============================================

CREATE POLICY "us_reportes_select_own" ON us_reportes
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "us_reportes_insert_own" ON us_reportes
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "us_reportes_update_own" ON us_reportes
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "us_reportes_delete_own" ON us_reportes
    FOR DELETE USING (auth.uid() = user_id);

CREATE POLICY "us_reportes_service_role" ON us_reportes
    FOR ALL USING (auth.jwt() ->> 'role' = 'service_role');
