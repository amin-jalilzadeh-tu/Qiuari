-- ============================================
-- ELECTRICAL GRID HIERARCHY - STEPS 4-8 (UPDATED)
-- Building Connections with Connection Types
-- Simplified: All buildings → LV, with MV capability flags
-- ============================================

-- Clean up existing tables
DROP TABLE IF EXISTS amin_grid.tlip_building_connections CASCADE;
DROP TABLE IF EXISTS amin_grid.tlip_building_connection_points CASCADE;
DROP TABLE IF EXISTS amin_grid.tlip_segment_connections CASCADE;
DROP TABLE IF EXISTS amin_grid.tlip_grid_summary CASCADE;
DROP VIEW IF EXISTS amin_grid.v_tlip_grid_overview CASCADE;
DROP VIEW IF EXISTS amin_grid.v_tlip_connection_types CASCADE;
DROP VIEW IF EXISTS amin_grid.v_tlip_problematic_connections CASCADE;

-- Verify required tables exist
DO $$
DECLARE
    v_building_count INTEGER;
    v_segment_count INTEGER;
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables 
                   WHERE table_schema = 'amin_grid' 
                   AND table_name = 'tlip_cable_segments') THEN
        RAISE EXCEPTION 'Table tlip_cable_segments does not exist. Please run Step 1 first.';
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables 
                   WHERE table_schema = 'amin_grid' 
                   AND table_name = 'tlip_buildings_1_deducted') THEN
        RAISE EXCEPTION 'Table tlip_buildings_1_deducted does not exist.';
    END IF;
    
    SELECT COUNT(*) INTO v_building_count 
    FROM amin_grid.tlip_buildings_1_deducted 
    WHERE area > 10 AND pand_geom IS NOT NULL;
    
    SELECT COUNT(*) INTO v_segment_count 
    FROM amin_grid.tlip_cable_segments;
    
    RAISE NOTICE 'Starting building connections...';
    RAISE NOTICE '  Buildings to process: %', v_building_count;
    RAISE NOTICE '  Cable segments available: %', v_segment_count;
END $$;

-- ============================================
-- 4. Create Building Connections Table (UPDATED)
-- ============================================

CREATE TABLE amin_grid.tlip_building_connections (
    building_id BIGINT PRIMARY KEY,
    -- Building properties
    building_area NUMERIC,
    building_height NUMERIC,
    building_function VARCHAR(50),
    building_type VARCHAR(100),
    building_geom GEOMETRY(GEOMETRY, 28992),
    building_centroid GEOMETRY(POINT, 28992),
    -- LV Connection details (all buildings connect to LV)
    connected_voltage VARCHAR(10) DEFAULT 'LV',
    connected_group_id VARCHAR(50),
    connected_segment_id BIGINT,
    connection_distance_m NUMERIC,
    connection_type VARCHAR(20), -- ENDED/ENTERED/CROSSED/BY_DISTANCE/TOO_FAR
    -- MV capability flags
    is_mv_capable BOOLEAN,
    has_mv_nearby BOOLEAN,
    nearest_mv_distance_m NUMERIC,
    nearest_mv_segment_id BIGINT,
    -- Problem detection
    is_problematic BOOLEAN,
    connection_reason TEXT,
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Step 4.1: Insert building connections with connection type detection
INSERT INTO amin_grid.tlip_building_connections
WITH building_analysis AS (
    SELECT 
        ogc_fid as building_id,
        area as building_area,
        COALESCE(height, gem_hoogte, 3.0) as building_height,
        building_function,
        COALESCE(non_residential_type, residential_type, 'unknown') as building_type,
        pand_geom as building_geom,
        ST_Centroid(pand_geom) as building_centroid,
        -- MV capability flag (simple criteria)
        CASE 
            WHEN building_function = 'non_residential' AND area > 3000 THEN TRUE
            ELSE FALSE
        END as is_mv_capable
    FROM amin_grid.tlip_buildings_1_deducted
    WHERE area > 10  -- Minimum building size
        AND pand_geom IS NOT NULL
        AND ST_IsValid(pand_geom)
),
-- Find closest LV connections for all buildings
lv_connections AS (
    SELECT DISTINCT ON (b.building_id)
        b.building_id,
        s.segment_id,
        s.group_id,
        s.segment_geom,
        ST_Distance(b.building_centroid, s.segment_geom) as distance,
        -- Check if cable ends in building
        CASE 
            WHEN ST_Within(s.start_point, b.building_geom) OR ST_Within(s.end_point, b.building_geom) 
            THEN 'ENDED'
            -- Check if cable intersects building
            WHEN ST_Intersects(s.segment_geom, b.building_geom) 
            THEN CASE 
                WHEN ST_Crosses(s.segment_geom, b.building_geom) THEN 'CROSSED'
                ELSE 'ENTERED'
            END
            -- Otherwise by distance
            WHEN ST_Distance(b.building_centroid, s.segment_geom) > 150 
            THEN 'TOO_FAR'
            ELSE 'BY_DISTANCE'
        END as connection_type
    FROM building_analysis b
    CROSS JOIN LATERAL (
        SELECT segment_id, group_id, segment_geom, start_point, end_point
        FROM amin_grid.tlip_cable_segments
        WHERE voltage_level = 'LV'
            AND ST_DWithin(segment_geom, b.building_centroid, 500)
        ORDER BY ST_Distance(segment_geom, b.building_centroid)
        LIMIT 1
    ) s
),
-- Find closest MV connections (for flagging only)
mv_connections AS (
    SELECT DISTINCT ON (b.building_id)
        b.building_id,
        s.segment_id as mv_segment_id,
        ST_Distance(b.building_centroid, s.segment_geom) as mv_distance
    FROM building_analysis b
    CROSS JOIN LATERAL (
        SELECT segment_id, segment_geom
        FROM amin_grid.tlip_cable_segments
        WHERE voltage_level = 'MV'
            AND ST_DWithin(segment_geom, b.building_centroid, 100)
        ORDER BY ST_Distance(segment_geom, b.building_centroid)
        LIMIT 1
    ) s
    WHERE b.is_mv_capable = TRUE  -- Only check MV for capable buildings
)
SELECT 
    b.building_id,
    b.building_area,
    b.building_height,
    b.building_function,
    b.building_type,
    b.building_geom,
    b.building_centroid,
    -- All buildings connect to LV
    'LV' as connected_voltage,
    lvc.group_id as connected_group_id,
    lvc.segment_id as connected_segment_id,
    lvc.distance as connection_distance_m,
    lvc.connection_type,
    -- MV capability flags
    b.is_mv_capable,
    CASE WHEN mvc.mv_distance IS NOT NULL THEN TRUE ELSE FALSE END as has_mv_nearby,
    mvc.mv_distance as nearest_mv_distance_m,
    mvc.mv_segment_id as nearest_mv_segment_id,
    -- Problem detection
    CASE WHEN lvc.connection_type = 'TOO_FAR' THEN TRUE ELSE FALSE END as is_problematic,
    -- Connection reason
    CASE
        WHEN lvc.connection_type = 'ENDED' 
        THEN 'Cable endpoint in building'
        WHEN lvc.connection_type = 'ENTERED' 
        THEN 'Cable enters building'
        WHEN lvc.connection_type = 'CROSSED' 
        THEN 'Cable crosses through building'
        WHEN lvc.connection_type = 'TOO_FAR' 
        THEN 'WARNING: Distance > 150m (' || ROUND(lvc.distance::numeric, 0) || 'm)'
        WHEN b.is_mv_capable AND mvc.mv_distance IS NOT NULL
        THEN 'LV connection (MV-capable, MV available at ' || ROUND(mvc.mv_distance::numeric, 0) || 'm)'
        WHEN b.is_mv_capable 
        THEN 'LV connection (MV-capable, no MV nearby)'
        WHEN b.building_function = 'residential' 
        THEN 'Residential - Standard LV'
        ELSE 'Standard LV connection at ' || ROUND(lvc.distance::numeric, 0) || 'm'
    END as connection_reason
FROM building_analysis b
LEFT JOIN lv_connections lvc ON b.building_id = lvc.building_id
LEFT JOIN mv_connections mvc ON b.building_id = mvc.building_id
WHERE lvc.segment_id IS NOT NULL  -- Must have LV connection
    AND lvc.distance < 1000;  -- Maximum connection distance

CREATE INDEX idx_tlip_building_conn_id ON amin_grid.tlip_building_connections(building_id);
CREATE INDEX idx_tlip_building_conn_segment ON amin_grid.tlip_building_connections(connected_segment_id);
CREATE INDEX idx_tlip_building_conn_group ON amin_grid.tlip_building_connections(connected_group_id);
CREATE INDEX idx_tlip_building_conn_type ON amin_grid.tlip_building_connections(connection_type);
CREATE INDEX idx_tlip_building_conn_mv_capable ON amin_grid.tlip_building_connections(is_mv_capable);
CREATE INDEX idx_tlip_building_conn_problematic ON amin_grid.tlip_building_connections(is_problematic);
CREATE INDEX idx_tlip_building_conn_geom ON amin_grid.tlip_building_connections USING GIST(building_geom);

-- ============================================
-- 5-6. Create Connection Points on Lines (UPDATED)
-- ============================================

CREATE TABLE amin_grid.tlip_building_connection_points (
    connection_point_id BIGSERIAL PRIMARY KEY,
    building_id BIGINT REFERENCES amin_grid.tlip_building_connections(building_id),
    segment_id BIGINT REFERENCES amin_grid.tlip_cable_segments(segment_id),
    group_id VARCHAR(50),
    voltage_level VARCHAR(10),
    connection_type VARCHAR(20),
    -- Connection point geometry
    point_on_line GEOMETRY(POINT, 28992),
    connection_line GEOMETRY(LINESTRING, 28992),
    -- Additional metadata
    distance_along_segment NUMERIC,
    segment_fraction NUMERIC,
    connection_distance_m NUMERIC,
    is_direct_connection BOOLEAN,  -- TRUE if cable physically connects to building
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create connection points with type information
INSERT INTO amin_grid.tlip_building_connection_points (
    building_id, segment_id, group_id, voltage_level, connection_type,
    point_on_line, connection_line, 
    distance_along_segment, segment_fraction, connection_distance_m,
    is_direct_connection
)
SELECT 
    bc.building_id,
    bc.connected_segment_id,
    bc.connected_group_id,
    bc.connected_voltage,
    bc.connection_type,
    -- Connection point depends on type
    CASE 
        WHEN bc.connection_type IN ('ENDED', 'ENTERED', 'CROSSED') 
        THEN ST_ClosestPoint(cs.segment_geom, bc.building_centroid)
        ELSE ST_ClosestPoint(cs.segment_geom, bc.building_centroid)
    END as point_on_line,
    -- Create line from building to connection point
    ST_MakeLine(
        bc.building_centroid,
        ST_ClosestPoint(cs.segment_geom, bc.building_centroid)
    ) as connection_line,
    -- Calculate distance along segment
    ST_LineLocatePoint(cs.segment_geom, 
        ST_ClosestPoint(cs.segment_geom, bc.building_centroid)
    ) * ST_Length(cs.segment_geom) as distance_along_segment,
    -- Calculate fraction along segment (0-1)
    ST_LineLocatePoint(cs.segment_geom, 
        ST_ClosestPoint(cs.segment_geom, bc.building_centroid)
    ) as segment_fraction,
    bc.connection_distance_m,
    -- Direct connection if cable physically touches building
    CASE 
        WHEN bc.connection_type IN ('ENDED', 'ENTERED', 'CROSSED') THEN TRUE
        ELSE FALSE
    END as is_direct_connection
FROM amin_grid.tlip_building_connections bc
JOIN amin_grid.tlip_cable_segments cs 
    ON bc.connected_segment_id = cs.segment_id
WHERE bc.connected_segment_id IS NOT NULL;

CREATE INDEX idx_tlip_conn_points_building ON amin_grid.tlip_building_connection_points(building_id);
CREATE INDEX idx_tlip_conn_points_segment ON amin_grid.tlip_building_connection_points(segment_id);
CREATE INDEX idx_tlip_conn_points_group ON amin_grid.tlip_building_connection_points(group_id);
CREATE INDEX idx_tlip_conn_points_type ON amin_grid.tlip_building_connection_points(connection_type);
CREATE INDEX idx_tlip_conn_points_direct ON amin_grid.tlip_building_connection_points(is_direct_connection);
CREATE INDEX idx_tlip_conn_points_geom ON amin_grid.tlip_building_connection_points USING GIST(point_on_line);

-- ============================================
-- 7. Calculate Segment Connection Statistics (SIMPLIFIED)
-- ============================================

CREATE TABLE amin_grid.tlip_segment_connections (
    segment_id BIGINT PRIMARY KEY,
    group_id VARCHAR(50),
    voltage_level VARCHAR(10),
    -- Connection counts
    total_buildings INTEGER,
    residential_count INTEGER,
    non_residential_count INTEGER,
    mv_capable_count INTEGER,
    -- Connection type counts
    ended_count INTEGER,
    entered_count INTEGER,
    crossed_count INTEGER,
    by_distance_count INTEGER,
    too_far_count INTEGER,
    -- Distance statistics
    avg_connection_distance_m NUMERIC,
    max_connection_distance_m NUMERIC,
    min_connection_distance_m NUMERIC,
    -- Flags
    has_problematic_connections BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Calculate connection statistics per segment
INSERT INTO amin_grid.tlip_segment_connections
SELECT 
    cs.segment_id,
    cs.group_id,
    cs.voltage_level,
    -- Connection counts
    COUNT(DISTINCT bc.building_id) as total_buildings,
    COUNT(DISTINCT CASE WHEN bc.building_function = 'residential' 
                   THEN bc.building_id END) as residential_count,
    COUNT(DISTINCT CASE WHEN bc.building_function = 'non_residential' 
                   THEN bc.building_id END) as non_residential_count,
    COUNT(DISTINCT CASE WHEN bc.is_mv_capable = TRUE 
                   THEN bc.building_id END) as mv_capable_count,
    -- Connection type counts
    COUNT(CASE WHEN bc.connection_type = 'ENDED' THEN 1 END) as ended_count,
    COUNT(CASE WHEN bc.connection_type = 'ENTERED' THEN 1 END) as entered_count,
    COUNT(CASE WHEN bc.connection_type = 'CROSSED' THEN 1 END) as crossed_count,
    COUNT(CASE WHEN bc.connection_type = 'BY_DISTANCE' THEN 1 END) as by_distance_count,
    COUNT(CASE WHEN bc.connection_type = 'TOO_FAR' THEN 1 END) as too_far_count,
    -- Distance statistics
    AVG(bc.connection_distance_m) as avg_connection_distance_m,
    MAX(bc.connection_distance_m) as max_connection_distance_m,
    MIN(bc.connection_distance_m) as min_connection_distance_m,
    -- Flags
    BOOL_OR(bc.is_problematic) as has_problematic_connections,
    CURRENT_TIMESTAMP
FROM amin_grid.tlip_cable_segments cs
LEFT JOIN amin_grid.tlip_building_connections bc
    ON cs.segment_id = bc.connected_segment_id
GROUP BY cs.segment_id, cs.group_id, cs.voltage_level;

CREATE INDEX idx_tlip_segment_conn_segment ON amin_grid.tlip_segment_connections(segment_id);
CREATE INDEX idx_tlip_segment_conn_group ON amin_grid.tlip_segment_connections(group_id);
CREATE INDEX idx_tlip_segment_conn_voltage ON amin_grid.tlip_segment_connections(voltage_level);
CREATE INDEX idx_tlip_segment_conn_problematic ON amin_grid.tlip_segment_connections(has_problematic_connections);

-- ============================================
-- 8. Create Summary Tables and Views (UPDATED)
-- ============================================

-- Summary statistics table
CREATE TABLE amin_grid.tlip_grid_summary (
    summary_id SERIAL PRIMARY KEY,
    summary_type VARCHAR(50),
    metric_name VARCHAR(100),
    metric_value NUMERIC,
    metric_text TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Populate summary statistics (no load data)
INSERT INTO amin_grid.tlip_grid_summary (summary_type, metric_name, metric_value)
VALUES 
    -- Building counts
    ('buildings', 'total_buildings', (SELECT COUNT(*) FROM amin_grid.tlip_building_connections)),
    ('buildings', 'residential', (SELECT COUNT(*) FROM amin_grid.tlip_building_connections WHERE building_function = 'residential')),
    ('buildings', 'non_residential', (SELECT COUNT(*) FROM amin_grid.tlip_building_connections WHERE building_function = 'non_residential')),
    ('buildings', 'mv_capable', (SELECT COUNT(*) FROM amin_grid.tlip_building_connections WHERE is_mv_capable = TRUE)),
    ('buildings', 'mv_capable_with_mv_nearby', (SELECT COUNT(*) FROM amin_grid.tlip_building_connections WHERE is_mv_capable = TRUE AND has_mv_nearby = TRUE)),
    -- Connection types
    ('connection_types', 'ended', (SELECT COUNT(*) FROM amin_grid.tlip_building_connections WHERE connection_type = 'ENDED')),
    ('connection_types', 'entered', (SELECT COUNT(*) FROM amin_grid.tlip_building_connections WHERE connection_type = 'ENTERED')),
    ('connection_types', 'crossed', (SELECT COUNT(*) FROM amin_grid.tlip_building_connections WHERE connection_type = 'CROSSED')),
    ('connection_types', 'by_distance', (SELECT COUNT(*) FROM amin_grid.tlip_building_connections WHERE connection_type = 'BY_DISTANCE')),
    ('connection_types', 'too_far', (SELECT COUNT(*) FROM amin_grid.tlip_building_connections WHERE connection_type = 'TOO_FAR')),
    -- Problems
    ('problems', 'problematic_connections', (SELECT COUNT(*) FROM amin_grid.tlip_building_connections WHERE is_problematic = TRUE)),
    -- Segments
    ('segments', 'segments_with_buildings', (SELECT COUNT(*) FROM amin_grid.tlip_segment_connections WHERE total_buildings > 0)),
    ('segments', 'segments_with_problems', (SELECT COUNT(*) FROM amin_grid.tlip_segment_connections WHERE has_problematic_connections = TRUE)),
    -- Distance
    ('distance', 'avg_connection_m', (SELECT AVG(connection_distance_m) FROM amin_grid.tlip_building_connections)),
    ('distance', 'max_connection_m', (SELECT MAX(connection_distance_m) FROM amin_grid.tlip_building_connections));

-- Create comprehensive overview view
CREATE OR REPLACE VIEW amin_grid.v_tlip_grid_overview AS
SELECT 
    'Grid Overview' as report,
    (SELECT COUNT(*) FROM amin_grid.tlip_cable_segments) as total_segments,
    (SELECT COUNT(*) FROM amin_grid.tlip_connected_groups) as cable_groups,
    (SELECT COUNT(*) FROM amin_grid.tlip_building_connections) as connected_buildings,
    (SELECT COUNT(*) FROM amin_grid.tlip_building_connections WHERE is_mv_capable = TRUE) as mv_capable_buildings,
    (SELECT COUNT(*) FROM amin_grid.tlip_building_connections WHERE is_mv_capable = TRUE AND has_mv_nearby = TRUE) as mv_ready_buildings,
    (SELECT COUNT(*) FROM amin_grid.tlip_building_connections WHERE is_problematic = TRUE) as problematic_buildings,
    (SELECT COUNT(*) FROM amin_grid.tlip_building_connections WHERE connection_type IN ('ENDED', 'ENTERED', 'CROSSED')) as direct_connections,
    (SELECT ROUND(AVG(connection_distance_m)::numeric, 1) FROM amin_grid.tlip_building_connections) as avg_distance_m,
    (SELECT COUNT(*) FROM amin_grid.tlip_segment_connections WHERE total_buildings > 0) as segments_with_buildings;

-- Create connection types summary view
CREATE OR REPLACE VIEW amin_grid.v_tlip_connection_types AS
SELECT 
    connection_type,
    COUNT(*) as building_count,
    ROUND(AVG(connection_distance_m)::numeric, 1) as avg_distance_m,
    ROUND(MAX(connection_distance_m)::numeric, 1) as max_distance_m,
    COUNT(CASE WHEN is_mv_capable THEN 1 END) as mv_capable_count,
    COUNT(CASE WHEN is_problematic THEN 1 END) as problematic_count,
    ROUND(100.0 * COUNT(*) / NULLIF((SELECT COUNT(*) FROM amin_grid.tlip_building_connections), 0), 1) as percentage
FROM amin_grid.tlip_building_connections
GROUP BY connection_type
ORDER BY building_count DESC;

-- Create problematic connections view
CREATE OR REPLACE VIEW amin_grid.v_tlip_problematic_connections AS
SELECT 
    building_id,
    building_function,
    building_type,
    ROUND(building_area::numeric, 0) as area_m2,
    connection_type,
    ROUND(connection_distance_m::numeric, 1) as distance_m,
    is_mv_capable,
    has_mv_nearby,
    CASE 
        WHEN has_mv_nearby THEN ROUND(nearest_mv_distance_m::numeric, 1)
        ELSE NULL
    END as mv_distance_m,
    connection_reason
FROM amin_grid.tlip_building_connections
WHERE is_problematic = TRUE
ORDER BY connection_distance_m DESC;

-- ============================================
-- VERIFICATION AND REPORTING
-- ============================================

DO $$
DECLARE
    r RECORD;
    v_total_buildings INTEGER;
    v_connected_buildings INTEGER;
    v_mv_capable INTEGER;
    v_problematic INTEGER;
BEGIN
    RAISE NOTICE '============================================';
    RAISE NOTICE 'STEPS 4-8 COMPLETED: Building Connections';
    RAISE NOTICE '============================================';
    
    -- Get counts
    SELECT COUNT(*) INTO v_total_buildings 
    FROM amin_grid.tlip_buildings_1_deducted 
    WHERE area > 10 AND pand_geom IS NOT NULL;
    
    SELECT COUNT(*) INTO v_connected_buildings 
    FROM amin_grid.tlip_building_connections;
    
    SELECT COUNT(*) INTO v_mv_capable 
    FROM amin_grid.tlip_building_connections 
    WHERE is_mv_capable = TRUE;
    
    SELECT COUNT(*) INTO v_problematic 
    FROM amin_grid.tlip_building_connections 
    WHERE is_problematic = TRUE;
    
    RAISE NOTICE '';
    RAISE NOTICE 'Building Connection Summary:';
    RAISE NOTICE '  Total eligible buildings: %', v_total_buildings;
    RAISE NOTICE '  Connected buildings: % (%.1f%%)', 
                 v_connected_buildings, 
                 (v_connected_buildings::float / NULLIF(v_total_buildings, 0) * 100);
    RAISE NOTICE '  MV-capable buildings: % (%.1f%%)', 
                 v_mv_capable,
                 (v_mv_capable::float / NULLIF(v_connected_buildings, 0) * 100);
    RAISE NOTICE '  Problematic connections (>150m): % (%.1f%%)', 
                 v_problematic,
                 (v_problematic::float / NULLIF(v_connected_buildings, 0) * 100);
    
    -- Connection types summary
    RAISE NOTICE '';
    RAISE NOTICE 'Connection Types Distribution:';
    FOR r IN 
        SELECT 
            connection_type,
            COUNT(*) as count,
            ROUND(AVG(connection_distance_m)::numeric, 1) as avg_dist
        FROM amin_grid.tlip_building_connections
        GROUP BY connection_type
        ORDER BY count DESC
    LOOP
        RAISE NOTICE '  %: % buildings (avg %.1fm)', 
                     RPAD(r.connection_type, 12), r.count, r.avg_dist;
    END LOOP;
    
    -- MV capability analysis
    RAISE NOTICE '';
    RAISE NOTICE 'MV Capability Analysis:';
    SELECT 
        COUNT(CASE WHEN is_mv_capable AND has_mv_nearby THEN 1 END),
        COUNT(CASE WHEN is_mv_capable AND NOT has_mv_nearby THEN 1 END),
        COUNT(CASE WHEN NOT is_mv_capable AND has_mv_nearby THEN 1 END),
        COUNT(CASE WHEN NOT is_mv_capable AND NOT has_mv_nearby THEN 1 END)
    INTO r
    FROM amin_grid.tlip_building_connections;
    
    RAISE NOTICE '  MV-capable with MV nearby: %', r.count;
    RAISE NOTICE '  MV-capable without MV: %', r.count;
    RAISE NOTICE '  Not MV-capable with MV nearby: %', r.count;
    RAISE NOTICE '  Standard LV only: %', r.count;
    
    -- Top segments by connections
    RAISE NOTICE '';
    RAISE NOTICE 'Top 5 Segments by Building Count:';
    FOR r IN 
        SELECT 
            segment_id,
            voltage_level,
            total_buildings,
            too_far_count
        FROM amin_grid.tlip_segment_connections
        WHERE total_buildings > 0
        ORDER BY total_buildings DESC
        LIMIT 5
    LOOP
        RAISE NOTICE '  Segment % (%): % buildings (% problematic)', 
                     r.segment_id, r.voltage_level, r.total_buildings, r.too_far_count;
    END LOOP;
    
    -- Direct connections
    SELECT 
        COUNT(CASE WHEN connection_type = 'ENDED' THEN 1 END),
        COUNT(CASE WHEN connection_type = 'ENTERED' THEN 1 END),
        COUNT(CASE WHEN connection_type = 'CROSSED' THEN 1 END)
    INTO r
    FROM amin_grid.tlip_building_connections;
    
    RAISE NOTICE '';
    RAISE NOTICE 'Direct Cable-Building Connections:';
    RAISE NOTICE '  Cable ends in building: %', r.count;
    RAISE NOTICE '  Cable enters building: %', r.count;
    RAISE NOTICE '  Cable crosses building: %', r.count;
    
    RAISE NOTICE '';
    RAISE NOTICE 'Connection points created: %', 
                 (SELECT COUNT(*) FROM amin_grid.tlip_building_connection_points);
    
    -- Problem summary
    IF v_problematic > 0 THEN
        RAISE NOTICE '';
        RAISE NOTICE 'WARNING: % buildings have problematic connections (>150m)', v_problematic;
        RAISE NOTICE 'Check v_tlip_problematic_connections view for details';
    END IF;
    
    RAISE NOTICE '';
    RAISE NOTICE 'STEPS 4-8 COMPLETED SUCCESSFULLY';
END $$;

-- Final overview
SELECT * FROM amin_grid.v_tlip_grid_overview;

-- Show connection type distribution
SELECT * FROM amin_grid.v_tlip_connection_types;

-- Show sample of problematic connections if any exist
DO $$
DECLARE
    r RECORD;
    v_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO v_count FROM amin_grid.v_tlip_problematic_connections;
    
    IF v_count > 0 THEN
        RAISE NOTICE '';
        RAISE NOTICE 'Sample Problematic Connections (first 5):';
        FOR r IN 
            SELECT * FROM amin_grid.v_tlip_problematic_connections
            LIMIT 5
        LOOP
            RAISE NOTICE '  Building %: %m away, %', 
                         r.building_id, r.distance_m, r.building_type;
        END LOOP;
    END IF;
END $$;