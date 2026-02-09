-- Add columns to parts if they don't exist (e.g. table was created from older schema)
ALTER TABLE parts ADD COLUMN IF NOT EXISTS partselect_number VARCHAR(50);
ALTER TABLE parts ADD COLUMN IF NOT EXISTS manufacturer_part_number VARCHAR(100);
ALTER TABLE parts ADD COLUMN IF NOT EXISTS name VARCHAR(255);
ALTER TABLE parts ADD COLUMN IF NOT EXISTS url TEXT;
-- Allow name to be NULL if we're adding it and existing rows have no name
-- (optional: UPDATE parts SET name = part_number WHERE name IS NULL; then ALTER COLUMN name SET NOT NULL;)
CREATE INDEX IF NOT EXISTS idx_parts_partselect ON parts (partselect_number) WHERE partselect_number IS NOT NULL;
