-- ============================================
-- ELECTRICAL GRID HIERARCHY - STEPS 2-3 (COMPLETE)
-- Establish Hierarchical Connections Between Groups and Stations
-- Fixed: Recursive CTE type issue in view
-- ============================================

-- Clean up existing tables and views
DROP VIEW IF EXISTS amin_grid.v_tlip_hierarchy_tree CASCADE;
DROP TABLE IF EXISTS amin_grid.tlip_group_stations CASCADE;
DROP TABLE IF EXISTS amin_grid.tlip_group_hierarchy CASCADE;
DROP TABLE IF EXISTS amin_grid.tlip_voltage_transitions CASCADE;

-- Verify required tables exist
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables 
                   WHERE table_schema = 'amin_grid' 
                   AND table_name = 'tlip_cable_segments') THEN
        RAISE EXCEPTION 'Table tlip_cable_segments does not exist. Please run Step 1 first.';
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables 
                   WHERE table_schema = 'amin_grid' 
                   AND table_name = 'tlip_connected_groups') THEN
        RAISE EXCEPTION 'Table tlip_connected_groups does not exist. Please run Step 1 first.';
    END IF;
    
    RAISE NOTICE 'Required tables verified. Proceeding with Steps 2-3...';
END $$;

-- ============================================
-- 2.1 Create Group-Station Connections Table
-- ============================================

CREATE TABLE amin_grid.tlip_group_stations (
    connection_id BIGSERIAL PRIMARY KEY,
    group_id VARCHAR(50),
    voltage_level VARCHAR(10),
    station_type VARCHAR(50),
    station_fid BIGINT,
    station_geom GEOMETRY(GEOMETRY, 28992),
    connection_type VARCHAR(20),
    distance_m NUMERIC,
    confidence_score NUMERIC,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Connect LV groups to LV cabinets
INSERT INTO amin_grid.tlip_group_stations (
    group_id, voltage_level, station_type, station_fid, 
    station_geom, connection_type, distance_m, confidence_score
)
SELECT 
    g.group_id,
    g.voltage_level,
    'LV_CABINET' as station_type,
    c.fid as station_fid,
    c.clipped_geom as station_geom,
    CASE 
        WHEN ST_DWithin(g.merged_geom, c.clipped_geom, 1) THEN 'DIRECT'
        ELSE 'PROXIMITY'
    END as connection_type,
    ST_Distance(g.merged_geom, c.clipped_geom) as distance_m,
    CASE 
        WHEN ST_DWithin(g.merged_geom, c.clipped_geom, 1) THEN 0.95
        WHEN ST_Distance(g.merged_geom, c.clipped_geom) < 10 THEN 0.85
        WHEN ST_Distance(g.merged_geom, c.clipped_geom) < 25 THEN 0.70
        WHEN ST_Distance(g.merged_geom, c.clipped_geom) < 50 THEN 0.50
        ELSE 0.30
    END as confidence_score
FROM amin_grid.tlip_connected_groups g
CROSS JOIN LATERAL (
    SELECT fid, clipped_geom
    FROM amin_grid.tlip_laagspanningsverdeelkasten
    WHERE clipped_geom IS NOT NULL
        AND ST_DWithin(clipped_geom, g.merged_geom, 100)
    ORDER BY ST_Distance(clipped_geom, g.merged_geom)
    LIMIT 3
) c
WHERE g.voltage_level = 'LV';

-- Connect LV groups to transformers
INSERT INTO amin_grid.tlip_group_stations (
    group_id, voltage_level, station_type, station_fid, 
    station_geom, connection_type, distance_m, confidence_score
)
SELECT DISTINCT ON (g.group_id)
    g.group_id,
    g.voltage_level,
    'TRANSFORMER' as station_type,
    t.fid as station_fid,
    t.clipped_geom as station_geom,
    CASE 
        WHEN ST_DWithin(g.merged_geom, t.clipped_geom, 2) THEN 'DIRECT'
        ELSE 'PROXIMITY'
    END as connection_type,
    ST_Distance(g.merged_geom, t.clipped_geom) as distance_m,
    CASE 
        WHEN ST_DWithin(g.merged_geom, t.clipped_geom, 2) THEN 0.99
        WHEN ST_Distance(g.merged_geom, t.clipped_geom) < 20 THEN 0.90
        WHEN ST_Distance(g.merged_geom, t.clipped_geom) < 50 THEN 0.75
        WHEN ST_Distance(g.merged_geom, t.clipped_geom) < 100 THEN 0.60
        WHEN ST_Distance(g.merged_geom, t.clipped_geom) < 200 THEN 0.40
        ELSE 0.20
    END as confidence_score
FROM amin_grid.tlip_connected_groups g
CROSS JOIN LATERAL (
    SELECT fid, clipped_geom
    FROM amin_grid.tlip_middenspanningsinstallaties
    WHERE clipped_geom IS NOT NULL
        AND ST_DWithin(clipped_geom, g.merged_geom, 300)
    ORDER BY ST_Distance(clipped_geom, g.merged_geom)
    LIMIT 1
) t
WHERE g.voltage_level = 'LV'
ORDER BY g.group_id, ST_Distance(g.merged_geom, t.clipped_geom);

-- Connect MV groups to transformers
INSERT INTO amin_grid.tlip_group_stations (
    group_id, voltage_level, station_type, station_fid, 
    station_geom, connection_type, distance_m, confidence_score
)
SELECT 
    g.group_id,
    g.voltage_level,
    'TRANSFORMER' as station_type,
    t.fid as station_fid,
    t.clipped_geom as station_geom,
    CASE 
        WHEN ST_DWithin(g.merged_geom, t.clipped_geom, 2) THEN 'DIRECT'
        ELSE 'PROXIMITY'
    END as connection_type,
    ST_Distance(g.merged_geom, t.clipped_geom) as distance_m,
    CASE 
        WHEN ST_DWithin(g.merged_geom, t.clipped_geom, 2) THEN 0.99
        WHEN ST_Distance(g.merged_geom, t.clipped_geom) < 10 THEN 0.95
        WHEN ST_Distance(g.merged_geom, t.clipped_geom) < 30 THEN 0.85
        WHEN ST_Distance(g.merged_geom, t.clipped_geom) < 100 THEN 0.65
        ELSE 0.40
    END as confidence_score
FROM amin_grid.tlip_connected_groups g
CROSS JOIN LATERAL (
    SELECT fid, clipped_geom
    FROM amin_grid.tlip_middenspanningsinstallaties
    WHERE clipped_geom IS NOT NULL
        AND ST_DWithin(clipped_geom, g.merged_geom, 200)
    ORDER BY ST_Distance(clipped_geom, g.merged_geom)
    LIMIT 5
) t
WHERE g.voltage_level = 'MV';

-- Connect MV groups to substations
INSERT INTO amin_grid.tlip_group_stations (
    group_id, voltage_level, station_type, station_fid, 
    station_geom, connection_type, distance_m, confidence_score
)
SELECT DISTINCT ON (g.group_id)
    g.group_id,
    g.voltage_level,
    'SUBSTATION' as station_type,
    s.fid as station_fid,
    s.clipped_geom as station_geom,
    CASE 
        WHEN ST_DWithin(g.merged_geom, s.clipped_geom, 5) THEN 'DIRECT'
        ELSE 'PROXIMITY'
    END as connection_type,
    ST_Distance(g.merged_geom, s.clipped_geom) as distance_m,
    CASE 
        WHEN ST_DWithin(g.merged_geom, s.clipped_geom, 5) THEN 0.99
        WHEN ST_Distance(g.merged_geom, s.clipped_geom) < 50 THEN 0.90
        WHEN ST_Distance(g.merged_geom, s.clipped_geom) < 200 THEN 0.70
        WHEN ST_Distance(g.merged_geom, s.clipped_geom) < 500 THEN 0.50
        ELSE 0.30
    END as confidence_score
FROM amin_grid.tlip_connected_groups g
CROSS JOIN LATERAL (
    SELECT fid, clipped_geom
    FROM amin_grid.tlip_onderstations
    WHERE clipped_geom IS NOT NULL
    ORDER BY ST_Distance(clipped_geom, g.merged_geom)
    LIMIT 1
) s
WHERE g.voltage_level = 'MV'
    AND ST_Distance(g.merged_geom, s.clipped_geom) < 1000
ORDER BY g.group_id, ST_Distance(g.merged_geom, s.clipped_geom);

-- Connect HV groups to substations
INSERT INTO amin_grid.tlip_group_stations (
    group_id, voltage_level, station_type, station_fid, 
    station_geom, connection_type, distance_m, confidence_score
)
SELECT 
    g.group_id,
    g.voltage_level,
    'SUBSTATION' as station_type,
    s.fid as station_fid,
    s.clipped_geom as station_geom,
    CASE 
        WHEN ST_DWithin(g.merged_geom, s.clipped_geom, 10) THEN 'DIRECT'
        ELSE 'PROXIMITY'
    END as connection_type,
    ST_Distance(g.merged_geom, s.clipped_geom) as distance_m,
    CASE 
        WHEN ST_DWithin(g.merged_geom, s.clipped_geom, 10) THEN 0.99
        WHEN ST_Distance(g.merged_geom, s.clipped_geom) < 100 THEN 0.85
        WHEN ST_Distance(g.merged_geom, s.clipped_geom) < 500 THEN 0.65
        ELSE 0.40
    END as confidence_score
FROM amin_grid.tlip_connected_groups g
CROSS JOIN LATERAL (
    SELECT fid, clipped_geom
    FROM amin_grid.tlip_onderstations
    WHERE clipped_geom IS NOT NULL
    ORDER BY ST_Distance(clipped_geom, g.merged_geom)
    LIMIT 2
) s
WHERE g.voltage_level = 'HV'
    AND ST_Distance(g.merged_geom, s.clipped_geom) < 2000;

CREATE INDEX idx_tlip_group_stations_group ON amin_grid.tlip_group_stations(group_id);
CREATE INDEX idx_tlip_group_stations_station ON amin_grid.tlip_group_stations(station_fid, station_type);
CREATE INDEX idx_tlip_group_stations_voltage ON amin_grid.tlip_group_stations(voltage_level);

-- ============================================
-- 3.1 Create Group Hierarchy Table
-- ============================================

CREATE TABLE amin_grid.tlip_group_hierarchy (
    hierarchy_id BIGSERIAL PRIMARY KEY,
    child_group_id VARCHAR(50),
    child_voltage VARCHAR(10),
    parent_group_id VARCHAR(50),
    parent_voltage VARCHAR(10),
    connection_via VARCHAR(50),
    via_station_fid BIGINT,
    distance_m NUMERIC,
    confidence_score NUMERIC,
    hierarchy_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Connect LV to MV through transformers
INSERT INTO amin_grid.tlip_group_hierarchy (
    child_group_id, child_voltage, parent_group_id, parent_voltage,
    connection_via, via_station_fid, distance_m, confidence_score
)
WITH lv_to_mv AS (
    SELECT 
        lv.group_id as lv_group,
        lv.voltage_level as lv_voltage,
        lv.station_fid as transformer_fid,
        mv.group_id as mv_group,
        mv.voltage_level as mv_voltage,
        lv.distance_m + mv.distance_m as total_distance,
        (lv.confidence_score + mv.confidence_score) / 2 as avg_confidence
    FROM amin_grid.tlip_group_stations lv
    JOIN amin_grid.tlip_group_stations mv
        ON lv.station_fid = mv.station_fid
        AND lv.station_type = 'TRANSFORMER'
        AND mv.station_type = 'TRANSFORMER'
        AND lv.voltage_level = 'LV'
        AND mv.voltage_level = 'MV'
)
SELECT DISTINCT ON (lv_group)
    lv_group,
    lv_voltage,
    mv_group,
    mv_voltage,
    'TRANSFORMER',
    transformer_fid,
    total_distance,
    avg_confidence
FROM lv_to_mv
ORDER BY lv_group, avg_confidence DESC, total_distance;

-- Connect MV to HV through substations
INSERT INTO amin_grid.tlip_group_hierarchy (
    child_group_id, child_voltage, parent_group_id, parent_voltage,
    connection_via, via_station_fid, distance_m, confidence_score
)
WITH mv_to_hv AS (
    SELECT 
        mv.group_id as mv_group,
        mv.voltage_level as mv_voltage,
        mv.station_fid as substation_fid,
        hv.group_id as hv_group,
        hv.voltage_level as hv_voltage,
        mv.distance_m + hv.distance_m as total_distance,
        (mv.confidence_score + hv.confidence_score) / 2 as avg_confidence
    FROM amin_grid.tlip_group_stations mv
    JOIN amin_grid.tlip_group_stations hv
        ON mv.station_fid = hv.station_fid
        AND mv.station_type = 'SUBSTATION'
        AND hv.station_type = 'SUBSTATION'
        AND mv.voltage_level = 'MV'
        AND hv.voltage_level = 'HV'
)
SELECT DISTINCT ON (mv_group)
    mv_group,
    mv_voltage,
    hv_group,
    hv_voltage,
    'SUBSTATION',
    substation_fid,
    total_distance,
    avg_confidence
FROM mv_to_hv
ORDER BY mv_group, avg_confidence DESC, total_distance;

-- Update hierarchy paths
UPDATE amin_grid.tlip_group_hierarchy h1
SET hierarchy_path = h1.parent_group_id || ' -> ' || h1.child_group_id
WHERE h1.child_voltage = 'MV';

UPDATE amin_grid.tlip_group_hierarchy h1
SET hierarchy_path = COALESCE(
    (SELECT h2.hierarchy_path || ' -> ' || h1.child_group_id
     FROM amin_grid.tlip_group_hierarchy h2
     WHERE h2.child_group_id = h1.parent_group_id), 
    h1.parent_group_id || ' -> ' || h1.child_group_id
)
WHERE h1.child_voltage = 'LV';

CREATE INDEX idx_tlip_hierarchy_child ON amin_grid.tlip_group_hierarchy(child_group_id);
CREATE INDEX idx_tlip_hierarchy_parent ON amin_grid.tlip_group_hierarchy(parent_group_id);
CREATE INDEX idx_tlip_hierarchy_station ON amin_grid.tlip_group_hierarchy(via_station_fid);

-- ============================================
-- 3.2 Create Voltage Transitions Table
-- ============================================

CREATE TABLE amin_grid.tlip_voltage_transitions (
    transition_id BIGSERIAL PRIMARY KEY,
    from_voltage VARCHAR(10),
    to_voltage VARCHAR(10),
    transition_type VARCHAR(50),
    station_fid BIGINT,
    station_geom GEOMETRY(GEOMETRY, 28992),
    from_groups TEXT[],
    to_groups TEXT[],
    transition_capacity VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Transformer transitions (LV to MV)
INSERT INTO amin_grid.tlip_voltage_transitions (
    from_voltage, to_voltage, transition_type, station_fid, station_geom,
    from_groups, to_groups
)
SELECT 
    'LV',
    'MV',
    'TRANSFORMER',
    t.fid,
    t.clipped_geom,
    array_agg(DISTINCT lv.group_id) FILTER (WHERE lv.group_id IS NOT NULL),
    array_agg(DISTINCT mv.group_id) FILTER (WHERE mv.group_id IS NOT NULL)
FROM amin_grid.tlip_middenspanningsinstallaties t
LEFT JOIN amin_grid.tlip_group_stations lv
    ON t.fid = lv.station_fid 
    AND lv.station_type = 'TRANSFORMER'
    AND lv.voltage_level = 'LV'
LEFT JOIN amin_grid.tlip_group_stations mv
    ON t.fid = mv.station_fid
    AND mv.station_type = 'TRANSFORMER'
    AND mv.voltage_level = 'MV'
WHERE t.clipped_geom IS NOT NULL
GROUP BY t.fid, t.clipped_geom
HAVING (array_agg(DISTINCT lv.group_id) FILTER (WHERE lv.group_id IS NOT NULL)) IS NOT NULL
    OR (array_agg(DISTINCT mv.group_id) FILTER (WHERE mv.group_id IS NOT NULL)) IS NOT NULL;

-- Substation transitions (MV to HV)
INSERT INTO amin_grid.tlip_voltage_transitions (
    from_voltage, to_voltage, transition_type, station_fid, station_geom,
    from_groups, to_groups
)
SELECT 
    'MV',
    'HV',
    'SUBSTATION',
    s.fid,
    s.clipped_geom,
    array_agg(DISTINCT mv.group_id) FILTER (WHERE mv.group_id IS NOT NULL),
    array_agg(DISTINCT hv.group_id) FILTER (WHERE hv.group_id IS NOT NULL)
FROM amin_grid.tlip_onderstations s
LEFT JOIN amin_grid.tlip_group_stations mv
    ON s.fid = mv.station_fid
    AND mv.station_type = 'SUBSTATION'
    AND mv.voltage_level = 'MV'
LEFT JOIN amin_grid.tlip_group_stations hv
    ON s.fid = hv.station_fid
    AND hv.station_type = 'SUBSTATION'
    AND hv.voltage_level = 'HV'
WHERE s.clipped_geom IS NOT NULL
GROUP BY s.fid, s.clipped_geom
HAVING (array_agg(DISTINCT mv.group_id) FILTER (WHERE mv.group_id IS NOT NULL)) IS NOT NULL
    OR (array_agg(DISTINCT hv.group_id) FILTER (WHERE hv.group_id IS NOT NULL)) IS NOT NULL;

CREATE INDEX idx_tlip_transitions_station ON amin_grid.tlip_voltage_transitions(station_fid);
CREATE INDEX idx_tlip_transitions_type ON amin_grid.tlip_voltage_transitions(transition_type);

-- ============================================
-- Create Hierarchy View (FIXED)
-- ============================================

CREATE OR REPLACE VIEW amin_grid.v_tlip_hierarchy_tree AS
WITH RECURSIVE hierarchy_tree AS (
    -- Start with HV groups (root level) - explicitly cast path to TEXT
    SELECT 
        g.group_id,
        g.voltage_level,
        NULL::VARCHAR(50) as parent_group,
        0 as level,
        g.group_id::TEXT as path  -- Explicit cast to TEXT
    FROM amin_grid.tlip_connected_groups g
    WHERE g.voltage_level = 'HV'
        AND NOT EXISTS (
            SELECT 1 FROM amin_grid.tlip_group_hierarchy h
            WHERE h.child_group_id = g.group_id
        )
    
    UNION ALL
    
    -- Add child groups recursively
    SELECT 
        h.child_group_id,
        h.child_voltage,
        h.parent_group_id,
        ht.level + 1,
        (ht.path || ' -> ' || h.child_group_id)::TEXT  -- Ensure TEXT type
    FROM amin_grid.tlip_group_hierarchy h
    JOIN hierarchy_tree ht ON h.parent_group_id = ht.group_id
)
SELECT 
    level,
    voltage_level,
    group_id,
    parent_group,
    path
FROM hierarchy_tree
ORDER BY level, voltage_level DESC, group_id;

-- ============================================
-- VERIFICATION
-- ============================================

DO $$
DECLARE
    r RECORD;
    station_conn_count INTEGER;
    hierarchy_count INTEGER;
    transition_count INTEGER;
BEGIN
    RAISE NOTICE '============================================';
    RAISE NOTICE 'STEPS 2-3 VERIFICATION';
    RAISE NOTICE '============================================';
    
    -- Check table creation
    SELECT COUNT(*) INTO station_conn_count FROM amin_grid.tlip_group_stations;
    SELECT COUNT(*) INTO hierarchy_count FROM amin_grid.tlip_group_hierarchy;
    SELECT COUNT(*) INTO transition_count FROM amin_grid.tlip_voltage_transitions;
    
    RAISE NOTICE '';
    RAISE NOTICE 'Tables Created:';
    RAISE NOTICE '  tlip_group_stations: % records', station_conn_count;
    RAISE NOTICE '  tlip_group_hierarchy: % records', hierarchy_count;
    RAISE NOTICE '  tlip_voltage_transitions: % records', transition_count;
    
    -- Station connections summary
    RAISE NOTICE '';
    RAISE NOTICE 'Station Connections:';
    FOR r IN 
        SELECT 
            voltage_level,
            station_type,
            COUNT(*) as connections,
            ROUND(AVG(distance_m)::numeric, 1) as avg_dist
        FROM amin_grid.tlip_group_stations
        GROUP BY voltage_level, station_type
        ORDER BY voltage_level, station_type
    LOOP
        RAISE NOTICE '  % - %: % connections, avg dist: %m', 
                     r.voltage_level, r.station_type, r.connections, r.avg_dist;
    END LOOP;
    
    -- Hierarchy summary
    RAISE NOTICE '';
    RAISE NOTICE 'Hierarchy Connections:';
    FOR r IN 
        SELECT 
            child_voltage || ' -> ' || parent_voltage as transition,
            COUNT(*) as count
        FROM amin_grid.tlip_group_hierarchy
        GROUP BY child_voltage, parent_voltage
    LOOP
        RAISE NOTICE '  %: % connections', r.transition, r.count;
    END LOOP;
    
    -- Orphaned groups
    SELECT COUNT(*) INTO r
    FROM amin_grid.tlip_connected_groups g
    WHERE NOT EXISTS (
        SELECT 1 FROM amin_grid.tlip_group_hierarchy h
        WHERE g.group_id = h.child_group_id OR g.group_id = h.parent_group_id
    )
    AND NOT EXISTS (
        SELECT 1 FROM amin_grid.tlip_group_stations s
        WHERE g.group_id = s.group_id
    );
    
    IF r.count > 0 THEN
        RAISE NOTICE '';
        RAISE NOTICE 'Warning: % groups not connected to any station or hierarchy', r.count;
    END IF;
    
    RAISE NOTICE '';
    RAISE NOTICE 'STEPS 2-3 COMPLETED SUCCESSFULLY';
END $$;

-- Final summary query
SELECT 
    'STEPS 2-3 SUMMARY' as report,
    (SELECT COUNT(*) FROM amin_grid.tlip_group_stations) as station_connections,
    (SELECT COUNT(*) FROM amin_grid.tlip_group_hierarchy) as hierarchy_links,
    (SELECT COUNT(*) FROM amin_grid.tlip_voltage_transitions) as transition_points,
    (SELECT COUNT(DISTINCT child_group_id) FROM amin_grid.tlip_group_hierarchy) as groups_with_parents,
    (SELECT COUNT(DISTINCT parent_group_id) FROM amin_grid.tlip_group_hierarchy) as parent_groups;