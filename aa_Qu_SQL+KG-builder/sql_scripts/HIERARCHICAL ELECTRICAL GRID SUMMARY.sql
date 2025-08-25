-- ============================================
-- HIERARCHICAL ELECTRICAL GRID SUMMARY
-- MV Stations → LV Groups → Buildings
-- With rolled-up statistics at each level
-- ============================================

-- First, create the complete hierarchy mapping
DROP TABLE IF EXISTS temp_grid_hierarchy;

CREATE TEMP TABLE temp_grid_hierarchy AS
WITH 
-- MV to LV mapping through transformers
mv_lv_mapping AS (
    SELECT DISTINCT
        mv_gs.station_fid as mv_station_id,
        mv_gs.group_id as mv_group_id,
        lv_gs.group_id as lv_group_id
    FROM amin_grid.tlip_group_stations mv_gs
    JOIN amin_grid.tlip_group_stations lv_gs
        ON mv_gs.station_fid = lv_gs.station_fid
        AND mv_gs.station_type = 'TRANSFORMER'
        AND lv_gs.station_type = 'TRANSFORMER'
        AND mv_gs.voltage_level = 'MV'
        AND lv_gs.voltage_level = 'LV'
),
-- Add building data
building_data AS (
    SELECT 
        bc.*,
        b.age_range,
        b.meestvoorkomendelabel as energy_label,
        b.woningtype as housing_type,
        b.wijknaam as district_name,
        b.buurtnaam as neighborhood_name
    FROM amin_grid.tlip_building_connections bc
    LEFT JOIN amin_grid.tlip_buildings_1_deducted b
        ON bc.building_id = b.ogc_fid
)
SELECT 
    m.mv_station_id,
    m.mv_group_id,
    m.lv_group_id,
    b.building_id,
    b.building_function,
    b.building_type,
    b.building_area,
    b.connection_type,
    b.connection_distance_m,
    b.is_mv_capable,
    b.has_mv_nearby,
    b.is_problematic,
    b.age_range,
    b.energy_label,
    b.housing_type,
    b.district_name,
    b.neighborhood_name
FROM mv_lv_mapping m
LEFT JOIN building_data b
    ON m.lv_group_id = b.connected_group_id;

-- ============================================
-- 1. MV STATION LEVEL SUMMARY
-- ============================================

SELECT '========== MV STATION LEVEL SUMMARY ==========' as report_section;

WITH mv_summary AS (
    SELECT 
        mv_station_id,
        -- Network counts
        COUNT(DISTINCT mv_group_id) as mv_groups,
        COUNT(DISTINCT lv_group_id) as lv_groups,
        COUNT(DISTINCT building_id) as total_buildings,
        -- Building function
        COUNT(DISTINCT CASE WHEN building_function = 'residential' THEN building_id END) as residential,
        COUNT(DISTINCT CASE WHEN building_function = 'non_residential' THEN building_id END) as non_residential,
        -- Building types (top categories)
        COUNT(DISTINCT CASE WHEN building_type = 'vrijstaand' THEN building_id END) as vrijstaand,
        COUNT(DISTINCT CASE WHEN building_type = 'twee_onder_1_kap' THEN building_id END) as twee_onder_1_kap,
        COUNT(DISTINCT CASE WHEN building_type = 'rijtjeswoning' THEN building_id END) as rijtjeswoning,
        COUNT(DISTINCT CASE WHEN building_type = 'appartement' THEN building_id END) as appartement,
        -- Connection types
        COUNT(CASE WHEN connection_type = 'ENDED' THEN 1 END) as ended,
        COUNT(CASE WHEN connection_type = 'ENTERED' THEN 1 END) as entered,
        COUNT(CASE WHEN connection_type = 'CROSSED' THEN 1 END) as crossed,
        COUNT(CASE WHEN connection_type = 'BY_DISTANCE' THEN 1 END) as by_distance,
        COUNT(CASE WHEN connection_type = 'TOO_FAR' THEN 1 END) as too_far,
        -- Age distribution
        COUNT(CASE WHEN age_range = '< 1945' THEN 1 END) as age_pre_1945,
        COUNT(CASE WHEN age_range = '1945-1975' THEN 1 END) as age_1945_1975,
        COUNT(CASE WHEN age_range = '1975-1990' THEN 1 END) as age_1975_1990,
        COUNT(CASE WHEN age_range = '1990-2005' THEN 1 END) as age_1990_2005,
        COUNT(CASE WHEN age_range = '2005-2015' THEN 1 END) as age_2005_2015,
        COUNT(CASE WHEN age_range = '> 2015' THEN 1 END) as age_post_2015,
        -- MV capability
        COUNT(DISTINCT CASE WHEN is_mv_capable THEN building_id END) as mv_capable,
        COUNT(DISTINCT CASE WHEN is_problematic THEN building_id END) as problematic,
        -- Area and distance
        ROUND(SUM(building_area)::numeric, 0) as total_area_m2,
        ROUND(AVG(building_area)::numeric, 0) as avg_area_m2,
        ROUND(AVG(connection_distance_m)::numeric, 1) as avg_distance_m,
        -- Energy labels
        COUNT(CASE WHEN energy_label IN ('A', 'A+', 'A++', 'A+++', 'A++++') THEN 1 END) as label_a,
        COUNT(CASE WHEN energy_label = 'B' THEN 1 END) as label_b,
        COUNT(CASE WHEN energy_label = 'C' THEN 1 END) as label_c,
        COUNT(CASE WHEN energy_label IN ('D', 'E', 'F', 'G') THEN 1 END) as label_defg
    FROM temp_grid_hierarchy
    GROUP BY mv_station_id
)
SELECT 
    mv_station_id,
    lv_groups as "LV Groups",
    total_buildings as "Total Buildings",
    residential || '/' || non_residential as "Res/NonRes",
    ROUND(100.0 * residential / NULLIF(total_buildings, 0), 0) || '%' as "% Res",
    mv_capable as "MV Capable",
    problematic as "Problem",
    ROUND(avg_area_m2, 0) as "Avg m²",
    ROUND(avg_distance_m, 1) as "Avg Dist",
    ended || '/' || entered || '/' || crossed || '/' || by_distance || '/' || too_far as "Connection Types (E/En/C/D/F)",
    age_pre_1945 || '/' || age_1945_1975 || '/' || age_1975_1990 || '/' || age_1990_2005 || '/' || age_2005_2015 || '/' || age_post_2015 as "Age Distribution",
    label_a || '/' || label_b || '/' || label_c || '/' || label_defg as "Energy (A/B/C/D-G)"
FROM mv_summary
ORDER BY total_buildings DESC
LIMIT 20;

-- ============================================
-- 2. LV GROUP LEVEL SUMMARY (Top 30 by building count)
-- ============================================

SELECT '========== LV GROUP LEVEL SUMMARY ==========' as report_section;

WITH lv_summary AS (
    SELECT 
        lv_group_id,
        mv_station_id,
        -- Building counts
        COUNT(DISTINCT building_id) as total_buildings,
        COUNT(DISTINCT CASE WHEN building_function = 'residential' THEN building_id END) as residential,
        COUNT(DISTINCT CASE WHEN building_function = 'non_residential' THEN building_id END) as non_residential,
        -- Connection types
        COUNT(CASE WHEN connection_type = 'ENDED' THEN 1 END) as ended,
        COUNT(CASE WHEN connection_type = 'ENTERED' THEN 1 END) as entered,
        COUNT(CASE WHEN connection_type = 'CROSSED' THEN 1 END) as crossed,
        COUNT(CASE WHEN connection_type = 'BY_DISTANCE' THEN 1 END) as by_distance,
        COUNT(CASE WHEN connection_type = 'TOO_FAR' THEN 1 END) as too_far,
        -- Age summary
        MODE() WITHIN GROUP (ORDER BY age_range) as dominant_age,
        -- MV capability
        COUNT(DISTINCT CASE WHEN is_mv_capable THEN building_id END) as mv_capable,
        COUNT(DISTINCT CASE WHEN is_problematic THEN building_id END) as problematic,
        -- Area and distance
        ROUND(SUM(building_area)::numeric, 0) as total_area_m2,
        ROUND(AVG(building_area)::numeric, 0) as avg_area_m2,
        ROUND(AVG(connection_distance_m)::numeric, 1) as avg_distance_m,
        ROUND(MAX(connection_distance_m)::numeric, 1) as max_distance_m,
        -- Energy
        MODE() WITHIN GROUP (ORDER BY energy_label) as common_energy_label,
        -- Location
        MODE() WITHIN GROUP (ORDER BY district_name) as district,
        MODE() WITHIN GROUP (ORDER BY building_type) as common_building_type
    FROM temp_grid_hierarchy
    WHERE building_id IS NOT NULL
    GROUP BY lv_group_id, mv_station_id
)
SELECT 
    lv_group_id as "LV Group",
    mv_station_id as "MV Station",
    total_buildings as "Buildings",
    residential || '/' || non_residential as "Res/NonRes",
    mv_capable as "MV Cap",
    problematic as "Prob",
    ROUND(avg_area_m2, 0) as "Avg m²",
    ROUND(avg_distance_m, 1) || '/' || ROUND(max_distance_m, 1) as "Avg/Max Dist",
    ended || '/' || entered || '/' || crossed || '/' || by_distance || '/' || too_far as "Connections (E/En/C/D/F)",
    dominant_age as "Main Age",
    common_energy_label as "Main Label",
    common_building_type as "Main Type",
    district as "District"
FROM lv_summary
ORDER BY total_buildings DESC
LIMIT 30;

-- ============================================
-- 3. MV STATION DETAILED BREAKDOWN (Top 10 stations)
-- ============================================

SELECT '========== DETAILED MV STATION BREAKDOWN ==========' as report_section;

WITH top_mv_stations AS (
    SELECT mv_station_id
    FROM temp_grid_hierarchy
    GROUP BY mv_station_id
    ORDER BY COUNT(DISTINCT building_id) DESC
    LIMIT 10
)
SELECT 
    h.mv_station_id as "MV Station",
    h.lv_group_id as "LV Group",
    COUNT(DISTINCT h.building_id) as "Buildings",
    COUNT(CASE WHEN h.building_function = 'residential' THEN 1 END) as "Res",
    COUNT(CASE WHEN h.building_function = 'non_residential' THEN 1 END) as "NonRes",
    COUNT(CASE WHEN h.is_mv_capable THEN 1 END) as "MV Cap",
    COUNT(CASE WHEN h.is_problematic THEN 1 END) as "Prob",
    ROUND(AVG(h.building_area)::numeric, 0) as "Avg m²",
    ROUND(AVG(h.connection_distance_m)::numeric, 1) as "Avg Dist",
    STRING_AGG(DISTINCT h.connection_type, ', ') as "Conn Types",
    MODE() WITHIN GROUP (ORDER BY h.age_range) as "Main Age",
    MODE() WITHIN GROUP (ORDER BY h.building_type) as "Main Type"
FROM temp_grid_hierarchy h
JOIN top_mv_stations t ON h.mv_station_id = t.mv_station_id
WHERE h.building_id IS NOT NULL
GROUP BY h.mv_station_id, h.lv_group_id
ORDER BY h.mv_station_id, COUNT(DISTINCT h.building_id) DESC;

-- ============================================
-- 4. ROLLUP SUMMARY - MV TO LV TO BUILDINGS
-- ============================================

SELECT '========== HIERARCHICAL ROLLUP SUMMARY ==========' as report_section;

SELECT 
    CASE 
        WHEN GROUPING(mv_station_id) = 1 THEN 'TOTAL GRID'
        ELSE 'MV Station ' || mv_station_id::text
    END as "Level",
    COUNT(DISTINCT lv_group_id) as "LV Groups",
    COUNT(DISTINCT building_id) as "Buildings",
    COUNT(DISTINCT CASE WHEN building_function = 'residential' THEN building_id END) as "Residential",
    COUNT(DISTINCT CASE WHEN building_function = 'non_residential' THEN building_id END) as "Non-Residential",
    COUNT(DISTINCT CASE WHEN is_mv_capable THEN building_id END) as "MV Capable",
    COUNT(DISTINCT CASE WHEN is_problematic THEN building_id END) as "Problematic",
    ROUND(AVG(building_area)::numeric, 0) as "Avg Area m²",
    ROUND(AVG(connection_distance_m)::numeric, 1) as "Avg Distance m"
FROM temp_grid_hierarchy
WHERE building_id IS NOT NULL
GROUP BY ROLLUP(mv_station_id)
ORDER BY GROUPING(mv_station_id), COUNT(DISTINCT building_id) DESC
LIMIT 25;

-- ============================================
-- 5. MV STATION COMPARISON MATRIX
-- ============================================

SELECT '========== MV STATION COMPARISON MATRIX ==========' as report_section;

SELECT 
    mv_station_id as "MV Station",
    COUNT(DISTINCT lv_group_id) as "LV Groups",
    COUNT(DISTINCT building_id) as "Total",
    ROUND(100.0 * COUNT(DISTINCT CASE WHEN building_function = 'residential' THEN building_id END) / 
          NULLIF(COUNT(DISTINCT building_id), 0), 0) as "% Res",
    ROUND(100.0 * COUNT(DISTINCT CASE WHEN is_mv_capable THEN building_id END) / 
          NULLIF(COUNT(DISTINCT building_id), 0), 0) as "% MV Cap",
    ROUND(100.0 * COUNT(DISTINCT CASE WHEN is_problematic THEN building_id END) / 
          NULLIF(COUNT(DISTINCT building_id), 0), 0) as "% Prob",
    ROUND(100.0 * COUNT(CASE WHEN connection_type IN ('ENDED', 'ENTERED', 'CROSSED') THEN 1 END) / 
          NULLIF(COUNT(*), 0), 0) as "% Direct",
    ROUND(100.0 * COUNT(CASE WHEN age_range IN ('< 1945', '1945-1975') THEN 1 END) / 
          NULLIF(COUNT(*), 0), 0) as "% Old",
    ROUND(100.0 * COUNT(CASE WHEN energy_label IN ('A', 'A+', 'A++', 'A+++', 'A++++', 'B', 'C') THEN 1 END) / 
          NULLIF(COUNT(*), 0), 0) as "% Efficient"
FROM temp_grid_hierarchy
WHERE building_id IS NOT NULL
GROUP BY mv_station_id
HAVING COUNT(DISTINCT building_id) > 50
ORDER BY COUNT(DISTINCT building_id) DESC
LIMIT 20;

-- ============================================
-- 6. LV GROUPS PER MV STATION WITH BUILDING SUMMARIES
-- ============================================

SELECT '========== LV GROUPS PER MV STATION ==========' as report_section;

WITH lv_per_mv AS (
    SELECT 
        mv_station_id,
        lv_group_id,
        COUNT(DISTINCT building_id) as buildings,
        COUNT(CASE WHEN building_function = 'residential' THEN 1 END) as res,
        COUNT(CASE WHEN building_function = 'non_residential' THEN 1 END) as non_res,
        COUNT(CASE WHEN is_mv_capable THEN 1 END) as mv_cap,
        COUNT(CASE WHEN is_problematic THEN 1 END) as prob,
        ROUND(AVG(connection_distance_m)::numeric, 1) as avg_dist
    FROM temp_grid_hierarchy
    WHERE building_id IS NOT NULL
    GROUP BY mv_station_id, lv_group_id
),
mv_totals AS (
    SELECT 
        mv_station_id,
        COUNT(DISTINCT lv_group_id) as total_lv_groups,
        SUM(buildings) as total_buildings
    FROM lv_per_mv
    GROUP BY mv_station_id
)
SELECT 
    l.mv_station_id as "MV Station",
    l.lv_group_id as "LV Group",
    l.buildings as "Bldgs",
    l.res || '/' || l.non_res as "R/NR",
    l.mv_cap as "MV",
    l.prob as "P",
    l.avg_dist as "Dist",
    ROUND(100.0 * l.buildings / m.total_buildings, 1) || '%' as "% of MV Total",
    '(' || m.total_lv_groups || ' LV, ' || m.total_buildings || ' bldgs)' as "MV Total"
FROM lv_per_mv l
JOIN mv_totals m ON l.mv_station_id = m.mv_station_id
WHERE m.total_buildings > 100
ORDER BY l.mv_station_id, l.buildings DESC
LIMIT 50;

-- ============================================
-- 7. SUMMARY STATISTICS BY HIERARCHY LEVEL
-- ============================================

SELECT '========== SUMMARY BY HIERARCHY LEVEL ==========' as report_section;

WITH level_stats AS (
    SELECT 
        'Per MV Station' as level,
        COUNT(DISTINCT mv_station_id) as units,
        ROUND(AVG(lv_per_mv), 1) as avg_lv_groups,
        ROUND(AVG(buildings_per_mv), 0) as avg_buildings,
        ROUND(AVG(res_pct), 1) as avg_res_pct,
        ROUND(AVG(mv_cap_pct), 1) as avg_mv_cap_pct,
        ROUND(AVG(prob_pct), 1) as avg_prob_pct
    FROM (
        SELECT 
            mv_station_id,
            COUNT(DISTINCT lv_group_id) as lv_per_mv,
            COUNT(DISTINCT building_id) as buildings_per_mv,
            100.0 * COUNT(DISTINCT CASE WHEN building_function = 'residential' THEN building_id END) / 
                NULLIF(COUNT(DISTINCT building_id), 0) as res_pct,
            100.0 * COUNT(DISTINCT CASE WHEN is_mv_capable THEN building_id END) / 
                NULLIF(COUNT(DISTINCT building_id), 0) as mv_cap_pct,
            100.0 * COUNT(DISTINCT CASE WHEN is_problematic THEN building_id END) / 
                NULLIF(COUNT(DISTINCT building_id), 0) as prob_pct
        FROM temp_grid_hierarchy
        GROUP BY mv_station_id
    ) mv_level
    UNION ALL
    SELECT 
        'Per LV Group' as level,
        COUNT(DISTINCT lv_group_id) as units,
        NULL as avg_lv_groups,
        ROUND(AVG(buildings_per_lv), 0) as avg_buildings,
        ROUND(AVG(res_pct), 1) as avg_res_pct,
        ROUND(AVG(mv_cap_pct), 1) as avg_mv_cap_pct,
        ROUND(AVG(prob_pct), 1) as avg_prob_pct
    FROM (
        SELECT 
            lv_group_id,
            COUNT(DISTINCT building_id) as buildings_per_lv,
            100.0 * COUNT(DISTINCT CASE WHEN building_function = 'residential' THEN building_id END) / 
                NULLIF(COUNT(DISTINCT building_id), 0) as res_pct,
            100.0 * COUNT(DISTINCT CASE WHEN is_mv_capable THEN building_id END) / 
                NULLIF(COUNT(DISTINCT building_id), 0) as mv_cap_pct,
            100.0 * COUNT(DISTINCT CASE WHEN is_problematic THEN building_id END) / 
                NULLIF(COUNT(DISTINCT building_id), 0) as prob_pct
        FROM temp_grid_hierarchy
        WHERE building_id IS NOT NULL
        GROUP BY lv_group_id
    ) lv_level
)
SELECT 
    level as "Hierarchy Level",
    units as "Total Units",
    COALESCE(avg_lv_groups::text, 'N/A') as "Avg LV Groups",
    avg_buildings as "Avg Buildings",
    avg_res_pct || '%' as "Avg % Residential",
    avg_mv_cap_pct || '%' as "Avg % MV Capable",
    avg_prob_pct || '%' as "Avg % Problematic"
FROM level_stats;

-- ============================================
-- 8. GRID TOTALS
-- ============================================

SELECT '========== GRID TOTALS ==========' as report_section;

SELECT 
    COUNT(DISTINCT mv_station_id) as "MV Stations",
    COUNT(DISTINCT mv_group_id) as "MV Groups",
    COUNT(DISTINCT lv_group_id) as "LV Groups",
    COUNT(DISTINCT building_id) as "Total Buildings",
    COUNT(DISTINCT CASE WHEN building_function = 'residential' THEN building_id END) as "Residential",
    COUNT(DISTINCT CASE WHEN building_function = 'non_residential' THEN building_id END) as "Non-Residential",
    COUNT(DISTINCT CASE WHEN is_mv_capable THEN building_id END) as "MV Capable",
    COUNT(DISTINCT CASE WHEN is_problematic THEN building_id END) as "Problematic",
    ROUND(AVG(connection_distance_m)::numeric, 1) as "Avg Distance (m)"
FROM temp_grid_hierarchy;

-- Clean up
DROP TABLE IF EXISTS temp_grid_hierarchy;