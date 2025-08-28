-- ============================================
-- MV/LV DISTRICT ANALYSIS FOR ENERGY COMMUNITIES & INTERVENTIONS
-- Analysis organized by electrical hierarchy (MV → LV → Districts)
-- ============================================

DROP TABLE IF EXISTS temp_mv_lv_district_analysis;
DROP TABLE IF EXISTS temp_mv_district_metrics;
DROP TABLE IF EXISTS temp_lv_district_metrics;

-- ============================================
-- 1. CREATE BASE DATA WITH MV/LV/DISTRICT HIERARCHY
-- ============================================

CREATE TEMP TABLE temp_mv_lv_district_analysis AS
WITH mv_lv_mapping AS (
    -- Get MV to LV mapping through transformers
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
building_enriched AS (
    SELECT 
        bc.*,
        b.age_range,
        b.meestvoorkomendelabel as energy_label,
        b.woningtype as housing_type,
        b.wijknaam as district_name,
        b.buurtnaam as neighborhood_name,
        b.bouwjaar as building_year,
        -- Simplified energy label
        CASE 
            WHEN b.meestvoorkomendelabel IN ('A', 'A+', 'A++', 'A+++', 'A++++') THEN 'A'
            WHEN b.meestvoorkomendelabel = 'B' THEN 'B'
            WHEN b.meestvoorkomendelabel = 'C' THEN 'C'
            WHEN b.meestvoorkomendelabel = 'D' THEN 'D'
            WHEN b.meestvoorkomendelabel = 'E' THEN 'E'
            WHEN b.meestvoorkomendelabel = 'F' THEN 'F'
            WHEN b.meestvoorkomendelabel = 'G' THEN 'G'
            ELSE 'Unknown'
        END as energy_label_simple
    FROM amin_grid.tlip_building_connections bc
    LEFT JOIN amin_grid.tlip_buildings_1_deducted b
        ON bc.building_id = b.ogc_fid
)
SELECT 
    m.mv_station_id,
    m.mv_group_id,
    m.lv_group_id,
    b.*
FROM mv_lv_mapping m
LEFT JOIN building_enriched b
    ON m.lv_group_id = b.connected_group_id
WHERE b.district_name IS NOT NULL;

-- ============================================
-- 2. MV STATION LEVEL METRICS BY DISTRICT
-- ============================================

CREATE TEMP TABLE temp_mv_district_metrics AS
WITH mv_district_stats AS (
    SELECT 
        mv_station_id,
        district_name,
        -- Scale
        COUNT(DISTINCT lv_group_id) as lv_groups_in_district,
        COUNT(DISTINCT building_id) as total_buildings,
        ROUND(SUM(building_area)::numeric, 0) as total_area_m2,
        
        -- Building diversity
        COUNT(DISTINCT building_function) as function_types,
        COUNT(DISTINCT building_type) as unique_building_types,
        COUNT(DISTINCT CASE WHEN building_function = 'residential' THEN building_id END) as residential_count,
        COUNT(DISTINCT CASE WHEN building_function = 'non_residential' THEN building_id END) as non_residential_count,
        
        -- Specific building types for temporal complementarity
        COUNT(DISTINCT CASE WHEN building_type IN ('Office', 'office', 'Kantoor') THEN building_id END) as office_count,
        COUNT(DISTINCT CASE WHEN building_type IN ('Retail', 'retail', 'Winkel') THEN building_id END) as retail_count,
        COUNT(DISTINCT CASE WHEN building_type IN ('Education', 'education', 'School') THEN building_id END) as education_count,
        COUNT(DISTINCT CASE WHEN building_type IN ('Industrial', 'industrial') THEN building_id END) as industrial_count,
        
        -- Size diversity
        ROUND(STDDEV(building_area)::numeric, 0) as stddev_building_area,
        COUNT(CASE WHEN building_area < 100 THEN 1 END) as small_buildings,
        COUNT(CASE WHEN building_area > 1000 THEN 1 END) as large_buildings,
        
        -- Age diversity
        COUNT(DISTINCT age_range) as unique_age_ranges,
        COUNT(CASE WHEN age_range IN ('< 1945', '1945-1975') THEN 1 END) as old_buildings,
        COUNT(CASE WHEN age_range IN ('2005-2015', '> 2015') THEN 1 END) as new_buildings,
        
        -- Energy efficiency
        COUNT(CASE WHEN energy_label_simple IN ('D', 'E', 'F', 'G') THEN 1 END) as poor_energy_labels,
        ROUND(100.0 * COUNT(CASE WHEN energy_label_simple IN ('D', 'E', 'F', 'G') THEN 1 END) / 
              NULLIF(COUNT(*), 0), 1) as pct_poor_labels,
        
        -- MV capability
        COUNT(CASE WHEN is_mv_capable THEN 1 END) as mv_capable_count,
        
        -- Connection quality
        ROUND(AVG(connection_distance_m)::numeric, 1) as avg_connection_distance,
        COUNT(CASE WHEN is_problematic THEN 1 END) as problematic_connections
        
    FROM temp_mv_lv_district_analysis
    GROUP BY mv_station_id, district_name
)
SELECT 
    *,
    -- Calculate diversity scores
    LEAST(10, unique_building_types * 1.5) as type_diversity_score,
    
    CASE 
        WHEN stddev_building_area > 500 THEN 10
        WHEN stddev_building_area > 300 THEN 8
        WHEN stddev_building_area > 150 THEN 6
        ELSE 3
    END as size_diversity_score,
    
    CASE 
        WHEN residential_count > 0 AND non_residential_count > 0 
             AND ABS(residential_count - non_residential_count) < total_buildings * 0.5 THEN 10
        WHEN residential_count > 0 AND non_residential_count > 0 THEN 6
        ELSE 2
    END as mix_balance_score,
    
    CASE 
        WHEN (office_count > 0 AND residential_count > 5) THEN 10
        WHEN (retail_count > 0 AND residential_count > 5) THEN 9
        WHEN (education_count > 0 AND residential_count > 5) THEN 8
        WHEN unique_age_ranges >= 4 THEN 6
        ELSE 3
    END as temporal_diversity_score,
    
    -- Intervention need scores
    CASE 
        WHEN pct_poor_labels > 70 THEN 10
        WHEN pct_poor_labels > 50 THEN 8
        WHEN pct_poor_labels > 30 THEN 6
        ELSE 3
    END as energy_intervention_need,
    
    CASE 
        WHEN old_buildings > total_buildings * 0.6 THEN 10
        WHEN old_buildings > total_buildings * 0.4 THEN 7
        WHEN old_buildings > total_buildings * 0.25 THEN 5
        ELSE 2
    END as age_intervention_need
    
FROM mv_district_stats;

-- ============================================
-- 3. LV GROUP LEVEL METRICS BY DISTRICT
-- ============================================

CREATE TEMP TABLE temp_lv_district_metrics AS
WITH lv_district_stats AS (
    SELECT 
        lv_group_id,
        mv_station_id,
        district_name,
        -- Scale
        COUNT(DISTINCT building_id) as total_buildings,
        
        -- Building diversity
        COUNT(DISTINCT building_type) as unique_building_types,
        COUNT(DISTINCT CASE WHEN building_function = 'residential' THEN building_id END) as residential_count,
        COUNT(DISTINCT CASE WHEN building_function = 'non_residential' THEN building_id END) as non_residential_count,
        
        -- Size diversity
        ROUND(STDDEV(building_area)::numeric, 0) as stddev_building_area,
        
        -- Age diversity
        COUNT(DISTINCT age_range) as unique_age_ranges,
        COUNT(CASE WHEN age_range IN ('< 1945', '1945-1975') THEN 1 END) as old_buildings,
        
        -- Energy efficiency
        COUNT(CASE WHEN energy_label_simple IN ('D', 'E', 'F', 'G') THEN 1 END) as poor_energy_labels,
        ROUND(100.0 * COUNT(CASE WHEN energy_label_simple IN ('D', 'E', 'F', 'G') THEN 1 END) / 
              NULLIF(COUNT(*), 0), 1) as pct_poor_labels,
        
        -- MV capability
        COUNT(CASE WHEN is_mv_capable THEN 1 END) as mv_capable_count
        
    FROM temp_mv_lv_district_analysis
    WHERE building_id IS NOT NULL
    GROUP BY lv_group_id, mv_station_id, district_name
)
SELECT 
    *,
    -- Simple diversity score for LV level
    ROUND((unique_building_types * 2 + 
           CASE WHEN residential_count > 0 AND non_residential_count > 0 THEN 5 ELSE 0 END +
           unique_age_ranges) / 2.0, 1) as lv_diversity_score,
    
    -- Intervention need
    ROUND((pct_poor_labels / 10 + 
           old_buildings * 10.0 / NULLIF(total_buildings, 1)), 1) as lv_intervention_need
    
FROM lv_district_stats;

-- ============================================
-- 4. REPORTS
-- ============================================

-- A. MV STATION SUMMARY WITH DISTRICT DIVERSITY
SELECT '========== MV STATIONS - DISTRICT DIVERSITY & INTERVENTION NEEDS ==========' as report;

WITH mv_summary AS (
    SELECT 
        mv_station_id,
        COUNT(DISTINCT district_name) as districts_served,
        SUM(total_buildings) as total_buildings,
        STRING_AGG(DISTINCT district_name, ', ' ORDER BY district_name) as district_list,
        
        -- Best district for EC
        MAX(type_diversity_score + size_diversity_score + mix_balance_score + temporal_diversity_score) as max_diversity_score,
        
        -- Worst district for intervention
        MAX(energy_intervention_need + age_intervention_need) as max_intervention_need,
        
        -- Average scores
        ROUND(AVG(type_diversity_score + size_diversity_score + mix_balance_score + temporal_diversity_score), 1) as avg_diversity,
        ROUND(AVG(energy_intervention_need + age_intervention_need), 1) as avg_intervention
        
    FROM temp_mv_district_metrics
    GROUP BY mv_station_id
)
SELECT 
    mv_station_id as "MV Station",
    districts_served as "Districts",
    total_buildings as "Buildings",
    ROUND(avg_diversity, 1) as "Avg Diversity",
    ROUND(avg_intervention, 1) as "Avg Int Need",
    ROUND(max_diversity_score, 1) as "Best EC Score",
    ROUND(max_intervention_need, 1) as "Max Int Need",
    LEFT(district_list, 50) || CASE WHEN LENGTH(district_list) > 50 THEN '...' ELSE '' END as "Districts Served"
FROM mv_summary
ORDER BY total_buildings DESC
LIMIT 15;

-- B. BEST MV-DISTRICT COMBINATIONS FOR ENERGY COMMUNITIES
SELECT '========== BEST MV-DISTRICT COMBINATIONS FOR ENERGY COMMUNITIES ==========' as report;

SELECT 
    mv_station_id as "MV Station",
    district_name as "District",
    total_buildings as "Buildings",
    lv_groups_in_district as "LV Groups",
    ROUND(type_diversity_score + size_diversity_score + mix_balance_score + temporal_diversity_score, 1) as "EC Score",
    residential_count || '/' || non_residential_count as "Res/NonRes",
    unique_building_types as "Types",
    ROUND(temporal_diversity_score, 1) as "Temporal",
    mv_capable_count as "MV Cap",
    CASE 
        WHEN (office_count > 0 AND residential_count > 5) THEN 'Office+Res'
        WHEN (retail_count > 0 AND residential_count > 5) THEN 'Retail+Res'
        WHEN (education_count > 0 AND residential_count > 5) THEN 'Edu+Res'
        WHEN unique_age_ranges >= 4 THEN 'Age Mix'
        ELSE 'Basic'
    END as "Pattern"
FROM temp_mv_district_metrics
WHERE total_buildings >= 20
ORDER BY (type_diversity_score + size_diversity_score + mix_balance_score + temporal_diversity_score) DESC
LIMIT 20;

-- C. MV-DISTRICT COMBINATIONS NEEDING INTERVENTION
SELECT '========== MV-DISTRICT COMBINATIONS NEEDING INTERVENTION ==========' as report;

SELECT 
    mv_station_id as "MV Station",
    district_name as "District",
    total_buildings as "Buildings",
    ROUND(energy_intervention_need + age_intervention_need, 1) as "Int Score",
    poor_energy_labels || ' (' || ROUND(pct_poor_labels, 0) || '%)' as "Poor Labels",
    old_buildings || ' (' || ROUND(100.0 * old_buildings / NULLIF(total_buildings, 0), 0) || '%)' as "Old Buildings",
    ROUND(energy_intervention_need, 1) as "Energy Need",
    ROUND(age_intervention_need, 1) as "Age Need"
FROM temp_mv_district_metrics
WHERE total_buildings >= 20
ORDER BY (energy_intervention_need + age_intervention_need) DESC
LIMIT 20;

-- D. LV GROUP ANALYSIS WITHIN DISTRICTS
SELECT '========== LV GROUPS - DISTRICT BREAKDOWN ==========' as report;

SELECT 
    lv_group_id as "LV Group",
    mv_station_id as "MV Station",
    district_name as "District",
    total_buildings as "Bldgs",
    residential_count || '/' || non_residential_count as "R/NR",
    unique_building_types as "Types",
    ROUND(lv_diversity_score, 1) as "Div Score",
    ROUND(lv_intervention_need, 1) as "Int Need",
    poor_energy_labels as "Poor EL",
    old_buildings as "Old",
    mv_capable_count as "MV"
FROM temp_lv_district_metrics
WHERE total_buildings >= 10
ORDER BY lv_diversity_score DESC
LIMIT 30;

-- E. MV STATIONS WITH MIXED DISTRICT OPPORTUNITIES
SELECT '========== MV STATIONS WITH MIXED OPPORTUNITIES (EC + INTERVENTION) ==========' as report;

WITH mv_opportunities AS (
    SELECT 
        mv_station_id,
        COUNT(DISTINCT district_name) as total_districts,
        COUNT(DISTINCT CASE WHEN (type_diversity_score + size_diversity_score + mix_balance_score + temporal_diversity_score) >= 25 
                            THEN district_name END) as high_ec_districts,
        COUNT(DISTINCT CASE WHEN (energy_intervention_need + age_intervention_need) >= 12 
                            THEN district_name END) as high_int_districts,
        COUNT(DISTINCT CASE WHEN (type_diversity_score + size_diversity_score + mix_balance_score + temporal_diversity_score) >= 25 
                            AND (energy_intervention_need + age_intervention_need) >= 12 
                            THEN district_name END) as both_high_districts,
        SUM(total_buildings) as total_buildings,
        ROUND(AVG(type_diversity_score + size_diversity_score + mix_balance_score + temporal_diversity_score), 1) as avg_ec_score,
        ROUND(AVG(energy_intervention_need + age_intervention_need), 1) as avg_int_score
    FROM temp_mv_district_metrics
    GROUP BY mv_station_id
)
SELECT 
    mv_station_id as "MV Station",
    total_districts as "Districts",
    total_buildings as "Buildings",
    high_ec_districts as "High EC",
    high_int_districts as "High Int",
    both_high_districts as "Both High",
    avg_ec_score as "Avg EC",
    avg_int_score as "Avg Int",
    CASE 
        WHEN both_high_districts > 0 THEN 'EXCELLENT ⭐'
        WHEN high_ec_districts > 0 AND high_int_districts > 0 THEN 'VERY GOOD'
        WHEN high_ec_districts > 0 OR high_int_districts > 0 THEN 'GOOD'
        ELSE 'MODERATE'
    END as "Opportunity"
FROM mv_opportunities
WHERE total_buildings >= 50
ORDER BY both_high_districts DESC, (high_ec_districts + high_int_districts) DESC
LIMIT 15;

-- F. DISTRICT COMPARISON ACROSS MV STATIONS
SELECT '========== SAME DISTRICT ACROSS DIFFERENT MV STATIONS ==========' as report;

WITH district_mv_comparison AS (
    SELECT 
        district_name,
        COUNT(DISTINCT mv_station_id) as mv_stations_serving,
        SUM(total_buildings) as total_buildings,
        STRING_AGG(DISTINCT mv_station_id::text, ', ' ORDER BY mv_station_id::text) as mv_station_list,
        ROUND(AVG(type_diversity_score + size_diversity_score + mix_balance_score + temporal_diversity_score), 1) as avg_ec_score,
        ROUND(AVG(energy_intervention_need + age_intervention_need), 1) as avg_int_score,
        ROUND(STDDEV(type_diversity_score + size_diversity_score + mix_balance_score + temporal_diversity_score), 1) as ec_score_variance
    FROM temp_mv_district_metrics
    GROUP BY district_name
    HAVING COUNT(DISTINCT mv_station_id) > 1
)
SELECT 
    district_name as "District",
    mv_stations_serving as "MV Stations",
    total_buildings as "Buildings",
    avg_ec_score as "Avg EC",
    avg_int_score as "Avg Int",
    ec_score_variance as "EC Variance",
    LEFT(mv_station_list, 30) || '...' as "MV Stations"
FROM district_mv_comparison
ORDER BY total_buildings DESC
LIMIT 15;

-- G. HIERARCHICAL ROLLUP: MV → LV → DISTRICT
SELECT '========== HIERARCHICAL SUMMARY: MV → LV → DISTRICT ==========' as report;

WITH hierarchy_summary AS (
    SELECT 
        mv_station_id,
        lv_group_id,
        district_name,
        COUNT(DISTINCT building_id) as buildings,
        COUNT(DISTINCT CASE WHEN building_function = 'residential' THEN building_id END) as res,
        COUNT(DISTINCT CASE WHEN building_function = 'non_residential' THEN building_id END) as non_res,
        COUNT(CASE WHEN energy_label_simple IN ('D', 'E', 'F', 'G') THEN 1 END) as poor_labels,
        ROUND(AVG(connection_distance_m), 1) as avg_dist
    FROM temp_mv_lv_district_analysis
    WHERE building_id IS NOT NULL
    GROUP BY CUBE(mv_station_id, lv_group_id, district_name)
)
SELECT 
    CASE 
        WHEN mv_station_id IS NULL THEN 'TOTAL GRID'
        WHEN lv_group_id IS NULL THEN 'MV ' || mv_station_id
        WHEN district_name IS NULL THEN '  LV ' || lv_group_id
        ELSE '    ' || district_name
    END as "Level",
    buildings as "Bldgs",
    res || '/' || non_res as "R/NR",
    poor_labels as "Poor EL",
    avg_dist as "Avg Dist"
FROM hierarchy_summary
WHERE NOT (mv_station_id IS NULL AND lv_group_id IS NOT NULL)
  AND NOT (lv_group_id IS NULL AND district_name IS NOT NULL)
  AND buildings > 0
ORDER BY 
    mv_station_id NULLS FIRST,
    lv_group_id NULLS FIRST,
    district_name NULLS FIRST
LIMIT 100;

-- H. EXPORT FOR DETAILED ANALYSIS
SELECT 
    mv_station_id,
    district_name,
    total_buildings,
    lv_groups_in_district,
    type_diversity_score,
    size_diversity_score,
    mix_balance_score,
    temporal_diversity_score,
    type_diversity_score + size_diversity_score + mix_balance_score + temporal_diversity_score as total_ec_score,
    energy_intervention_need,
    age_intervention_need,
    energy_intervention_need + age_intervention_need as total_int_score,
    residential_count,
    non_residential_count,
    unique_building_types,
    pct_poor_labels,
    mv_capable_count
FROM temp_mv_district_metrics
ORDER BY mv_station_id, (type_diversity_score + size_diversity_score + mix_balance_score + temporal_diversity_score) DESC;

-- Clean up
DROP TABLE IF EXISTS temp_mv_lv_district_analysis;
DROP TABLE IF EXISTS temp_mv_district_metrics;
DROP TABLE IF EXISTS temp_lv_district_metrics;










\*

## **Key Features of This MV/LV-Based Analysis:**

### **1. MV Station Level Analysis**
Shows for each MV station:
- Which districts it serves
- Average diversity and intervention scores across its districts
- Best district for EC potential
- Worst district needing intervention

### **2. MV-District Combinations**
Evaluates each MV station + district pair for:
- **EC Potential**: Diversity of buildings under that MV in that district
- **Intervention Need**: Energy efficiency problems in that area
- **Temporal Patterns**: Office+Res, Retail+Res, Education+Res combinations

### **3. LV Group Analysis**
Shows for each LV group:
- Which district it serves
- Building diversity within that LV group
- Intervention needs at the LV level

### **4. Key Insights Provided:**

**Best EC Opportunities**: 
- MV stations serving diverse districts
- MV-district pairs with high complementarity
- LV groups with good building mix

**Intervention Priorities**:
- MV-district combinations with poor energy labels
- LV groups with old buildings
- Concentrated areas needing renovation

**Mixed Opportunities (⭐)**:
- MV stations with BOTH high EC potential AND intervention needs
- These are your best targets for maximum impact!

### **5. Special Analysis Features:**

**Cross-Infrastructure**: 
- Shows when the same district is served by multiple MV stations
- Identifies split districts that might complicate EC formation

**Hierarchical Rollup**:
- Total grid → MV stations → LV groups → Districts
- Shows building distribution at each level

**Pattern Detection**:
- Identifies Office+Residential (day/night complementarity)
- Retail+Residential (day/evening complementarity)
- Education+Residential (school hours complementarity)

## **How to Use This Analysis:**

### **For Energy Community Formation:**
1. Look at "BEST MV-DISTRICT COMBINATIONS" table
2. Select MV-district pairs with EC Score ≥25
3. Check if they have good temporal patterns (Office+Res, etc.)
4. Verify sufficient scale (≥20 buildings)

### **For Intervention Planning:**
1. Check "MV-DISTRICT COMBINATIONS NEEDING INTERVENTION"
2. Focus on areas with Int Score ≥12
3. Prioritize based on number of buildings affected

### **For Maximum Impact:**
1. Use "MV STATIONS WITH MIXED OPPORTUNITIES" table
2. Target areas marked "EXCELLENT ⭐" or "VERY GOOD"
3. These allow both EC formation AND efficiency improvements

## **Key Advantages of This Approach:**

1. **Electrical Reality**: Energy communities must share infrastructure, so organizing by MV/LV makes more sense than pure geography

2. **Infrastructure Constraints**: Shows which districts can actually form communities together (same MV station)

3. **Grid Planning**: Helps utilities understand load diversity at the transformer level

4. **Investment Efficiency**: Identifies where grid upgrades would benefit the most diverse set of buildings

This analysis directly supports your GNN research by:
- Providing labeled examples of good/bad EC locations
- Showing real electrical constraints
- Quantifying diversity at the infrastructure level
- Identifying patterns that lead to complementarity

The MV-district combinations with high scores are your best candidates for detailed study!
*/