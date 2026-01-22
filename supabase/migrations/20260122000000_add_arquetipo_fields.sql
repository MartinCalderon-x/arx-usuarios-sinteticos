-- ============================================
-- Usuarios Sintéticos - Add arquetipo fields
-- Migration to add industria and nivel_digital
-- ============================================

-- Add new columns to us_arquetipos
ALTER TABLE us_arquetipos
ADD COLUMN IF NOT EXISTS industria TEXT,
ADD COLUMN IF NOT EXISTS nivel_digital TEXT;

-- Create index for common filters
CREATE INDEX IF NOT EXISTS idx_us_arquetipos_industria ON us_arquetipos(industria);
CREATE INDEX IF NOT EXISTS idx_us_arquetipos_nivel_digital ON us_arquetipos(nivel_digital);

-- Refresh PostgREST schema cache
NOTIFY pgrst, 'reload schema';
