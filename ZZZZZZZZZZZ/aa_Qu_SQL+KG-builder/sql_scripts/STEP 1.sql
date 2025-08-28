-- ============================================
-- ELECTRICAL GRID HIERARCHY - STEP 1 (WORKING VERSION)
-- Using LATERAL joins to avoid CASE statement issues
-- ============================================

-- Clean up existing tables
DROP TABLE IF EXISTS amin_grid.tlip_cable_segments CASCADE;
DROP TABLE IF EXISTS amin_grid.tlip_segment_endpoints CASCADE;
DROP TABLE IF EXISTS amin_grid.tlip_connected_groups CASCADE;
DROP TABLE IF EXISTS amin_grid.tlip_group_segments CASCADE;

-- ============================================
-- 1.1 Create Cable Segments Table
-- ============================================

CREATE TABLE amin_grid.tlip_cable_segments (
    segment_id BIGSERIAL PRIMARY KEY,
    original_fid BIGINT,
    voltage_level VARCHAR(10),
    segment_geom GEOMETRY(LINESTRING, 28992),
    start_point GEOMETRY(POINT, 28992),
    end_point GEOMETRY(POINT, 28992),
    length_m NUMERIC,
    group_id VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Process LV cables using LATERAL join
INSERT INTO amin_grid.tlip_cable_segments (original_fid, voltage_level, segment_geom, start_point, end_point, length_m)
SELECT 
    lv.fid as original_fid,
    'LV' as voltage_level,
    dumped.geom::geometry(LineString, 28992) as segment_geom,
    ST_StartPoint(dumped.geom) as start_point,
    ST_EndPoint(dumped.geom) as end_point,
    ST_Length(dumped.geom) as length_m
FROM 
    amin_grid.tlip_laagspanningskabels lv,
    LATERAL ST_Dump(lv.clipped_geom) AS dumped(path, geom)
WHERE 
    lv.clipped_geom IS NOT NULL 
    AND ST_IsValid(lv.clipped_geom)
    AND ST_GeometryType(dumped.geom) = 'ST_LineString'
    AND ST_Length(dumped.geom) > 0.01;

-- Process MV cables
INSERT INTO amin_grid.tlip_cable_segments (original_fid, voltage_level, segment_geom, start_point, end_point, length_m)
SELECT 
    mv.fid as original_fid,
    'MV' as voltage_level,
    dumped.geom::geometry(LineString, 28992) as segment_geom,
    ST_StartPoint(dumped.geom) as start_point,
    ST_EndPoint(dumped.geom) as end_point,
    ST_Length(dumped.geom) as length_m
FROM 
    amin_grid.tlip_middenspanningskabels mv,
    LATERAL ST_Dump(mv.clipped_geom) AS dumped(path, geom)
WHERE 
    mv.clipped_geom IS NOT NULL 
    AND ST_IsValid(mv.clipped_geom)
    AND ST_GeometryType(dumped.geom) = 'ST_LineString'
    AND ST_Length(dumped.geom) > 0.01;

-- Process HV cables
INSERT INTO amin_grid.tlip_cable_segments (original_fid, voltage_level, segment_geom, start_point, end_point, length_m)
SELECT 
    hv.fid as original_fid,
    'HV' as voltage_level,
    dumped.geom::geometry(LineString, 28992) as segment_geom,
    ST_StartPoint(dumped.geom) as start_point,
    ST_EndPoint(dumped.geom) as end_point,
    ST_Length(dumped.geom) as length_m
FROM 
    amin_grid.tlip_hoogspanningskabels hv,
    LATERAL ST_Dump(hv.clipped_geom) AS dumped(path, geom)
WHERE 
    hv.clipped_geom IS NOT NULL 
    AND ST_IsValid(hv.clipped_geom)
    AND ST_GeometryType(dumped.geom) = 'ST_LineString'
    AND ST_Length(dumped.geom) > 0.01;

-- Create indexes
CREATE INDEX idx_tlip_segments_geom ON amin_grid.tlip_cable_segments USING GIST(segment_geom);
CREATE INDEX idx_tlip_segments_start ON amin_grid.tlip_cable_segments USING GIST(start_point);
CREATE INDEX idx_tlip_segments_end ON amin_grid.tlip_cable_segments USING GIST(end_point);
CREATE INDEX idx_tlip_segments_voltage ON amin_grid.tlip_cable_segments(voltage_level);
CREATE INDEX idx_tlip_segments_group ON amin_grid.tlip_cable_segments(group_id);

-- Check counts
DO $$
DECLARE
    lv_count INTEGER;
    mv_count INTEGER;
    hv_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO lv_count FROM amin_grid.tlip_cable_segments WHERE voltage_level = 'LV';
    SELECT COUNT(*) INTO mv_count FROM amin_grid.tlip_cable_segments WHERE voltage_level = 'MV';
    SELECT COUNT(*) INTO hv_count FROM amin_grid.tlip_cable_segments WHERE voltage_level = 'HV';
    
    RAISE NOTICE 'Segments created - LV: %, MV: %, HV: %', lv_count, mv_count, hv_count;
END $$;

-- ============================================
-- 1.2 Create Endpoint Connectivity Table
-- ============================================

CREATE TABLE amin_grid.tlip_segment_endpoints (
    endpoint_id BIGSERIAL PRIMARY KEY,
    segment_id BIGINT REFERENCES amin_grid.tlip_cable_segments(segment_id),
    voltage_level VARCHAR(10),
    point_geom GEOMETRY(POINT, 28992),
    point_type VARCHAR(10),
    snap_tolerance NUMERIC DEFAULT 0.5
);

INSERT INTO amin_grid.tlip_segment_endpoints (segment_id, voltage_level, point_geom, point_type)
SELECT segment_id, voltage_level, start_point, 'START'
FROM amin_grid.tlip_cable_segments
UNION ALL
SELECT segment_id, voltage_level, end_point, 'END'
FROM amin_grid.tlip_cable_segments;

CREATE INDEX idx_tlip_endpoints_geom ON amin_grid.tlip_segment_endpoints USING GIST(point_geom);
CREATE INDEX idx_tlip_endpoints_voltage ON amin_grid.tlip_segment_endpoints(voltage_level);

-- ============================================
-- 1.3 Create Connected Components
-- ============================================

CREATE TEMP TABLE segment_adjacency AS
SELECT DISTINCT
    e1.segment_id as seg1_id,
    e2.segment_id as seg2_id,
    e1.voltage_level
FROM amin_grid.tlip_segment_endpoints e1
JOIN amin_grid.tlip_segment_endpoints e2
    ON e1.voltage_level = e2.voltage_level
    AND e1.segment_id < e2.segment_id
    AND ST_DWithin(e1.point_geom, e2.point_geom, 0.5);

CREATE INDEX idx_temp_adj_seg1 ON segment_adjacency(seg1_id);
CREATE INDEX idx_temp_adj_seg2 ON segment_adjacency(seg2_id);

-- Assign group IDs
DO $$
DECLARE
    v_group_id INTEGER;
    v_voltage VARCHAR;
    v_unassigned BIGINT;
    v_count INTEGER;
BEGIN
    FOREACH v_voltage IN ARRAY ARRAY['LV', 'MV', 'HV'] LOOP
        v_group_id := 1;
        v_count := 0;
        
        RAISE NOTICE 'Processing % voltage level...', v_voltage;
        
        LOOP
            SELECT segment_id INTO v_unassigned
            FROM amin_grid.tlip_cable_segments
            WHERE voltage_level = v_voltage AND group_id IS NULL
            LIMIT 1;
            
            EXIT WHEN v_unassigned IS NULL;
            
            WITH RECURSIVE connected AS (
                SELECT v_unassigned::BIGINT as segment_id
                UNION
                SELECT DISTINCT
                    CASE 
                        WHEN sa.seg1_id = c.segment_id THEN sa.seg2_id
                        ELSE sa.seg1_id
                    END::BIGINT
                FROM connected c
                JOIN segment_adjacency sa 
                    ON (sa.seg1_id = c.segment_id OR sa.seg2_id = c.segment_id)
                    AND sa.voltage_level = v_voltage
            )
            UPDATE amin_grid.tlip_cable_segments
            SET group_id = v_voltage || '_GROUP_' || LPAD(v_group_id::TEXT, 4, '0')
            WHERE segment_id IN (SELECT segment_id FROM connected);
            
            v_group_id := v_group_id + 1;
            v_count := v_count + 1;
            
            IF v_count % 10 = 0 THEN
                RAISE NOTICE '  % groups processed', v_count;
            END IF;
        END LOOP;
        
        RAISE NOTICE '% complete: % groups', v_voltage, v_count;
    END LOOP;
END $$;

DROP TABLE segment_adjacency;

-- ============================================
-- 1.4 Create Connected Groups Summary
-- ============================================

CREATE TABLE amin_grid.tlip_connected_groups (
    group_id VARCHAR(50) PRIMARY KEY,
    voltage_level VARCHAR(10),
    segment_count INTEGER,
    total_length_m NUMERIC,
    merged_geom GEOMETRY(GEOMETRY, 28992),
    bbox GEOMETRY(GEOMETRY, 28992),
    centroid GEOMETRY(POINT, 28992),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert with proper bbox handling
INSERT INTO amin_grid.tlip_connected_groups
SELECT 
    group_id,
    voltage_level,
    COUNT(*) as segment_count,
    SUM(length_m) as total_length_m,
    ST_Union(segment_geom) as merged_geom,
    -- Handle linear features that might not create a polygon envelope
    CASE 
        WHEN ST_GeometryType(ST_Envelope(ST_Union(segment_geom))) = 'ST_Polygon' 
        THEN ST_Envelope(ST_Union(segment_geom))
        ELSE ST_Buffer(ST_Envelope(ST_Union(segment_geom)), 0.1)
    END as bbox,
    ST_Centroid(ST_Union(segment_geom)) as centroid,
    CURRENT_TIMESTAMP
FROM amin_grid.tlip_cable_segments
WHERE group_id IS NOT NULL
GROUP BY group_id, voltage_level;

CREATE INDEX idx_tlip_groups_geom ON amin_grid.tlip_connected_groups USING GIST(merged_geom);
CREATE INDEX idx_tlip_groups_bbox ON amin_grid.tlip_connected_groups USING GIST(bbox);
CREATE INDEX idx_tlip_groups_voltage ON amin_grid.tlip_connected_groups(voltage_level);

-- ============================================
-- 1.5 Create Group-Segment Mapping
-- ============================================

CREATE TABLE amin_grid.tlip_group_segments (
    id BIGSERIAL PRIMARY KEY,
    group_id VARCHAR(50),
    segment_id BIGINT,
    segment_order INTEGER,
    FOREIGN KEY (group_id) REFERENCES amin_grid.tlip_connected_groups(group_id),
    FOREIGN KEY (segment_id) REFERENCES amin_grid.tlip_cable_segments(segment_id)
);

INSERT INTO amin_grid.tlip_group_segments (group_id, segment_id, segment_order)
SELECT 
    group_id,
    segment_id,
    ROW_NUMBER() OVER (PARTITION BY group_id ORDER BY segment_id)
FROM amin_grid.tlip_cable_segments
WHERE group_id IS NOT NULL;

CREATE INDEX idx_tlip_group_segments_group ON amin_grid.tlip_group_segments(group_id);
CREATE INDEX idx_tlip_group_segments_segment ON amin_grid.tlip_group_segments(segment_id);

-- ============================================
-- FINAL SUMMARY
-- ============================================

DO $$
DECLARE
    r RECORD;
BEGIN
    RAISE NOTICE '============================================';
    RAISE NOTICE 'STEP 1 COMPLETED';
    RAISE NOTICE '============================================';
    
    FOR r IN 
        SELECT 
            voltage_level,
            COUNT(DISTINCT group_id) as group_count,
            COUNT(*) as segment_count,
            ROUND(AVG(length_m)::numeric, 2) as avg_length,
            ROUND(SUM(length_m)::numeric, 2) as total_length
        FROM amin_grid.tlip_cable_segments
        WHERE group_id IS NOT NULL
        GROUP BY voltage_level
        ORDER BY voltage_level
    LOOP
        RAISE NOTICE '% Level: % groups, % segments, avg: %m, total: %m', 
                     r.voltage_level, r.group_count, r.segment_count, 
                     r.avg_length, r.total_length;
    END LOOP;
    
    SELECT COUNT(*) INTO r FROM amin_grid.tlip_connected_groups;
    RAISE NOTICE '';
    RAISE NOTICE 'Total connected groups: %', r.count;
    
    RAISE NOTICE '';
    RAISE NOTICE 'Top 5 Groups by Length:';
    FOR r IN 
        SELECT 
            group_id,
            segment_count,
            ROUND(total_length_m::numeric, 2) as length_m
        FROM amin_grid.tlip_connected_groups
        ORDER BY total_length_m DESC
        LIMIT 5
    LOOP
        RAISE NOTICE '  %: % segments, %m', r.group_id, r.segment_count, r.length_m;
    END LOOP;
END $$;

-- Create summary view
CREATE OR REPLACE VIEW amin_grid.v_tlip_group_summary AS
SELECT 
    g.group_id,
    g.voltage_level,
    g.segment_count,
    ROUND(g.total_length_m::numeric, 2) as total_length_m,
    COUNT(DISTINCT s.original_fid) as original_cable_count,
    ROUND(ST_XMin(g.bbox)::numeric, 2) as bbox_xmin,
    ROUND(ST_YMin(g.bbox)::numeric, 2) as bbox_ymin,
    ROUND(ST_XMax(g.bbox)::numeric, 2) as bbox_xmax,
    ROUND(ST_YMax(g.bbox)::numeric, 2) as bbox_ymax
FROM amin_grid.tlip_connected_groups g
JOIN amin_grid.tlip_group_segments gs ON g.group_id = gs.group_id
JOIN amin_grid.tlip_cable_segments s ON gs.segment_id = s.segment_id
GROUP BY g.group_id, g.voltage_level, g.segment_count, g.total_length_m, g.bbox
ORDER BY g.voltage_level, g.total_length_m DESC;

-- Final check
SELECT 
    'STEP 1 STATUS' as status,
    (SELECT COUNT(*) FROM amin_grid.tlip_cable_segments) as total_segments,
    (SELECT COUNT(DISTINCT group_id) FROM amin_grid.tlip_connected_groups) as total_groups,
    (SELECT COUNT(*) FROM amin_grid.tlip_connected_groups WHERE voltage_level = 'LV') as lv_groups,
    (SELECT COUNT(*) FROM amin_grid.tlip_connected_groups WHERE voltage_level = 'MV') as mv_groups,
    (SELECT COUNT(*) FROM amin_grid.tlip_connected_groups WHERE voltage_level = 'HV') as hv_groups;