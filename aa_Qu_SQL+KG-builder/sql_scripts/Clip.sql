-- first need to run this, and select building ogc fid and get info to insert in next step
SELECT 
    ogc_fid,
    ST_AsText(ST_Centroid(pand_geom)) as centroid,
    ST_XMin(pand_geom) as xmin,
    ST_YMin(pand_geom) as ymin,
    ST_XMax(pand_geom) as xmax,
    ST_YMax(pand_geom) as ymax
FROM amin.buildings_1_deducted
WHERE ogc_fid IN (4804870, 4794514);










-- Create a table for the new clipping box
CREATE TABLE IF NOT EXISTS amin_grid.tlip_box (
    id SERIAL PRIMARY KEY,
    geom geometry(Polygon, 28992)
);

-- Clear any existing tlip boxes
TRUNCATE amin_grid.tlip_box;

-- Insert the box polygon using the two new buildings as diagonal corners
-- Using centroids from buildings 4794514 and 4804870
INSERT INTO amin_grid.tlip_box (geom)
SELECT ST_MakeEnvelope(
    118998.07145900543,  -- xmin (from building 4804870 centroid)
    482179.62932332687,  -- ymin (from building 4804870 centroid)
    121436.52633285055,  -- xmax (from building 4794514 centroid)
    483907.6409493442,   -- ymax (from building 4794514 centroid)
    28992                -- SRID
);

-- Verify the box was created
SELECT 
    id,
    ST_AsText(geom) as box_wkt,
    ST_Area(geom) as area_m2
FROM amin_grid.tlip_box;

-- Drop existing tlip_ tables if they exist (optional - remove if you want to keep existing)
DROP TABLE IF EXISTS amin_grid.tlip_onderstations CASCADE;
DROP TABLE IF EXISTS amin_grid.tlip_middenspanningsinstallaties CASCADE;
DROP TABLE IF EXISTS amin_grid.tlip_laagspanningsverdeelkasten CASCADE;
DROP TABLE IF EXISTS amin_grid.tlip_middenspanningskabels CASCADE;
DROP TABLE IF EXISTS amin_grid.tlip_laagspanningskabels CASCADE;
DROP TABLE IF EXISTS amin_grid.tlip_hoogspanningskabels CASCADE;
DROP TABLE IF EXISTS amin_grid.tlip_buildings_1_deducted CASCADE;

-- 1. Clip onderstations with tlip_ prefix
CREATE TABLE amin_grid.tlip_onderstations AS
SELECT 
    o.*,
    ST_Intersection(o.geom, t.geom) as clipped_geom
FROM amin_grid.onderstations o
JOIN amin_grid.tlip_box t ON ST_Intersects(o.geom, t.geom);

-- 2. Clip middenspanningsinstallaties with tlip_ prefix
CREATE TABLE amin_grid.tlip_middenspanningsinstallaties AS
SELECT 
    m.*,
    ST_Intersection(m.geom, t.geom) as clipped_geom
FROM amin_grid.middenspanningsinstallaties m
JOIN amin_grid.tlip_box t ON ST_Intersects(m.geom, t.geom);

-- 3. Clip laagspanningsverdeelkasten with tlip_ prefix
CREATE TABLE amin_grid.tlip_laagspanningsverdeelkasten AS
SELECT 
    l.*,
    ST_Intersection(l.geom, t.geom) as clipped_geom
FROM amin_grid.laagspanningsverdeelkasten l
JOIN amin_grid.tlip_box t ON ST_Intersects(l.geom, t.geom);

-- 4. Clip middenspanningskabels with tlip_ prefix
CREATE TABLE amin_grid.tlip_middenspanningskabels AS
SELECT 
    m.*,
    ST_Intersection(m.geom, t.geom) as clipped_geom
FROM amin_grid.middenspanningskabels m
JOIN amin_grid.tlip_box t ON ST_Intersects(m.geom, t.geom);

-- 5. Clip laagspanningskabels with tlip_ prefix
CREATE TABLE amin_grid.tlip_laagspanningskabels AS
SELECT 
    l.*,
    ST_Intersection(l.geom, t.geom) as clipped_geom
FROM amin_grid.laagspanningskabels l
JOIN amin_grid.tlip_box t ON ST_Intersects(l.geom, t.geom);

-- 6. Clip hoogspanningskabels with tlip_ prefix
CREATE TABLE amin_grid.tlip_hoogspanningskabels AS
SELECT 
    h.*,
    ST_Intersection(h.geom, t.geom) as clipped_geom
FROM amin_grid.hoogspanningskabels h
JOIN amin_grid.tlip_box t ON ST_Intersects(h.geom, t.geom);

-- 7. Clip buildings_1_deducted with tlip_ prefix
CREATE TABLE amin_grid.tlip_buildings_1_deducted AS
SELECT 
    b.*,
    ST_Intersection(b.pand_geom, t.geom) as clipped_geom
FROM amin.buildings_1_deducted b
JOIN amin_grid.tlip_box t ON ST_Intersects(b.pand_geom, t.geom);

-- Check the row counts to see how many features were clipped
SELECT 'tlip_onderstations' as table_name, COUNT(*) as count FROM amin_grid.tlip_onderstations
UNION ALL
SELECT 'tlip_middenspanningsinstallaties', COUNT(*) FROM amin_grid.tlip_middenspanningsinstallaties
UNION ALL
SELECT 'tlip_laagspanningsverdeelkasten', COUNT(*) FROM amin_grid.tlip_laagspanningsverdeelkasten
UNION ALL
SELECT 'tlip_middenspanningskabels', COUNT(*) FROM amin_grid.tlip_middenspanningskabels
UNION ALL
SELECT 'tlip_laagspanningskabels', COUNT(*) FROM amin_grid.tlip_laagspanningskabels
UNION ALL
SELECT 'tlip_hoogspanningskabels', COUNT(*) FROM amin_grid.tlip_hoogspanningskabels
UNION ALL
SELECT 'tlip_buildings_1_deducted', COUNT(*) FROM amin_grid.tlip_buildings_1_deducted
ORDER BY table_name;