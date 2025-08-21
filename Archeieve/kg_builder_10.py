"""
Knowledge Graph Builder for Energy District Analysis - PRE-GNN VERSION
Creates the foundational KG with raw data, infrastructure, and potential
Complementarity and clustering will be added AFTER GNN analysis
"""

import pandas as pd
import numpy as np
from neo4j import GraphDatabase
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnergyKnowledgeGraphBuilder:
    """Build Knowledge Graph from energy district data - Pre-GNN version"""
    
    def __init__(self, uri: str, user: str, password: str):
        """Initialize Neo4j connection"""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.stats = {
            'nodes_created': {},
            'relationships_created': {},
            'processing_time': {}
        }
        logger.info(f"Connected to Neo4j at {uri}")
    
    def close(self):
        """Close database connection"""
        self.driver.close()
        logger.info("Neo4j connection closed")
    
    def clear_database(self):
        """Clear all nodes and relationships"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("Database cleared")
    
    # ============================================
    # STEP 1: SCHEMA SETUP
    # ============================================
    
    def create_schema(self):
        """Create constraints and indexes for pre-GNN KG"""
        logger.info("Creating schema constraints and indexes...")
        
        constraints = [
            # Infrastructure
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Substation) REQUIRE s.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (m:MV_Transformer) REQUIRE m.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (l:LV_Network) REQUIRE l.component_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (b:Building) REQUIRE b.ogc_fid IS UNIQUE",
            
            # Temporal
            "CREATE CONSTRAINT IF NOT EXISTS FOR (t:TimeSlot) REQUIRE t.slot_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:EnergyState) REQUIRE e.state_id IS UNIQUE",
            
            # Assets
            "CREATE CONSTRAINT IF NOT EXISTS FOR (sol:SolarSystem) REQUIRE sol.system_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (bat:BatterySystem) REQUIRE bat.system_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (hp:HeatPumpSystem) REQUIRE hp.system_id IS UNIQUE",
        ]
        
        indexes = [
            # Performance indexes
            "CREATE INDEX IF NOT EXISTS FOR (b:Building) ON (b.lv_component_id)",
            "CREATE INDEX IF NOT EXISTS FOR (b:Building) ON (b.building_function)",
            "CREATE INDEX IF NOT EXISTS FOR (b:Building) ON (b.has_solar)",
            "CREATE INDEX IF NOT EXISTS FOR (t:TimeSlot) ON (t.timestamp)",
            "CREATE INDEX IF NOT EXISTS FOR (t:TimeSlot) ON (t.hour_of_day)",
            "CREATE INDEX IF NOT EXISTS FOR (e:EnergyState) ON (e.building_id)",
            "CREATE INDEX IF NOT EXISTS FOR (e:EnergyState) ON (e.timeslot_id)",
        ]
        
        with self.driver.session() as session:
            for constraint in constraints:
                session.run(constraint)
            for index in indexes:
                session.run(index)
        
        logger.info("Schema created successfully")
    
    # ============================================
    # STEP 2: LOAD GRID INFRASTRUCTURE
    # ============================================
    
    def load_grid_infrastructure(self, data_path: str):
        """Load grid topology: Substation -> MV -> LV hierarchy"""
        logger.info("Loading grid infrastructure...")
        
        # Load data files
        lv_networks = pd.read_csv(f"{data_path}/lv_networks.csv")
        mv_transformers = pd.read_csv(f"{data_path}/mv_transformers.csv")
        
        with self.driver.session() as session:
            # Create Substation
            session.run("""
                CREATE (s:Substation {
                    id: 'SUB_001',
                    name: 'Main Substation',
                    x: 100000,
                    y: 450000,
                    capacity_mva: 50,
                    voltage_level: 'HV'
                })
            """)
            self.stats['nodes_created']['Substation'] = 1
            
            # Create MV Transformers
            for _, mv in mv_transformers.iterrows():
                session.run("""
                    CREATE (m:MV_Transformer {
                        id: $id,
                        x: $x,
                        y: $y,
                        capacity_kva: $capacity,
                        substation_id: $sub_id,
                        voltage_level: 'MV'
                    })
                """, 
                id=mv['id'],
                x=float(mv['x']),
                y=float(mv['y']),
                capacity=float(mv['capacity_kva']),
                sub_id=mv['substation_id']
                )
            self.stats['nodes_created']['MV_Transformer'] = len(mv_transformers)
            
            # Create LV Networks
            for _, lv in lv_networks.iterrows():
                session.run("""
                    CREATE (l:LV_Network {
                        component_id: $comp_id,
                        network_id: $net_id,
                        x: $x,
                        y: $y,
                        mv_transformer_id: $mv_id,
                        capacity_kva: $capacity,
                        voltage_level: 'LV'
                    })
                """,
                comp_id=int(lv['component_id']),
                net_id=lv['id'],
                x=float(lv['x']),
                y=float(lv['y']),
                mv_id=lv['mv_transformer_id'],
                capacity=float(lv['capacity_kva'])
                )
            self.stats['nodes_created']['LV_Network'] = len(lv_networks)
            
            # Create hierarchy relationships
            # MV Transformers -> Substation
            session.run("""
                MATCH (m:MV_Transformer), (s:Substation {id: m.substation_id})
                CREATE (m)-[:FEEDS_FROM {capacity_kva: m.capacity_kva}]->(s)
            """)
            
            # LV Networks -> MV Transformers
            result = session.run("""
                MATCH (l:LV_Network), (m:MV_Transformer {id: l.mv_transformer_id})
                CREATE (l)-[:FEEDS_FROM {capacity_kva: l.capacity_kva}]->(m)
                RETURN count(*) as count
            """)
            
            self.stats['relationships_created']['FEEDS_FROM'] = result.single()['count'] + len(mv_transformers)
            
        logger.info(f"Created {sum(self.stats['nodes_created'].values())} infrastructure nodes")
    
    # ============================================
    # STEP 3: LOAD AND ENHANCE BUILDINGS
    # ============================================
    
    def load_buildings(self, data_path: str):
        """Load buildings with derived features"""
        logger.info("Loading building data...")
        
        buildings = pd.read_csv(f"{data_path}/buildings.csv")
        
        # Add derived features
        buildings = self._calculate_building_features(buildings)
        
        with self.driver.session() as session:
            for _, building in buildings.iterrows():
                # Create building node with all features
                session.run("""
                    CREATE (b:Building {
                        ogc_fid: $ogc_fid,
                        x: $x,
                        y: $y,
                        building_function: $function,
                        residential_type: $res_type,
                        non_residential_type: $non_res_type,
                        area: $area,
                        height: $height,
                        age_range: $age,
                        building_orientation_cardinal: $orientation,
                        roof_area: $roof_area,
                        flat_roof_area: $flat_roof,
                        sloped_roof_area: $sloped_roof,
                        suitable_roof_area: $suitable_roof,
                        lv_component_id: $lv_id,
                        lv_network_id: $lv_net_id,
                        
                        // Derived features
                        energy_label: $energy_label,
                        insulation_quality: $insulation,
                        solar_potential: $solar_pot,
                        solar_capacity_kwp: $solar_kwp,
                        battery_readiness: $battery_ready,
                        electrification_feasibility: $elec_feasible,
                        expected_cop: $cop,
                        
                        // Current assets
                        has_solar: $has_solar,
                        has_battery: $has_battery,
                        has_heat_pump: $has_hp,
                        heating_system: $heating
                    })
                """,
                ogc_fid=int(building['ogc_fid']),
                x=float(building['x']),
                y=float(building['y']),
                function=building['building_function'],
                res_type=building['residential_type'] if pd.notna(building['residential_type']) else 'None',
                non_res_type=building['non_residential_type'] if pd.notna(building['non_residential_type']) else 'None',
                area=float(building['area']),
                height=float(building['height']),
                age=building['age_range'],
                orientation=building['building_orientation_cardinal'],
                roof_area=float(building['roof_area']),
                flat_roof=float(building['flat_roof_area']),
                sloped_roof=float(building['sloped_roof_area']),
                suitable_roof=float(building['suitable_roof_area']),
                lv_id=int(building['lv_component_id']),
                lv_net_id=building['lv_network_id'],
                energy_label=building['energy_label'],
                insulation=building['insulation_quality'],
                solar_pot=building['solar_potential'],
                solar_kwp=float(building['solar_capacity_kwp']),
                battery_ready=building['battery_readiness'],
                elec_feasible=building['electrification_feasibility'],
                cop=float(building['expected_cop']),
                has_solar=bool(building['has_solar']),
                has_battery=bool(building['has_battery']),
                has_hp=bool(building['has_heat_pump']),
                heating=building['heating_system']
                )
            
            # Create CONNECTED_TO relationships
            result = session.run("""
                MATCH (b:Building), (l:LV_Network {component_id: b.lv_component_id})
                WITH b, l, 
                     sqrt((b.x - l.x)^2 + (b.y - l.y)^2) as distance
                CREATE (b)-[:CONNECTED_TO {
                    distance_m: distance,
                    cable_type: 'underground'
                }]->(l)
                RETURN count(*) as count
            """)
            
            self.stats['nodes_created']['Building'] = len(buildings)
            self.stats['relationships_created']['CONNECTED_TO'] = result.single()['count']
            
        logger.info(f"Created {len(buildings)} building nodes with enhanced features")
    
    def _calculate_building_features(self, buildings: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived building features"""
        
        # Energy label based on age and type
        def get_energy_label(row):
            age_labels = {
                '<1945': 'F',
                '1945-1975': 'E',
                '1975-1995': 'D',
                '1995-2015': 'C',
                '>2015': 'B'
            }
            base_label = age_labels.get(row['age_range'], 'D')
            
            # Adjust for building type
            if row['building_function'] == 'non_residential':
                if row['non_residential_type'] == 'Office' and row['age_range'] == '>2015':
                    return 'A'
            return base_label
        
        buildings['energy_label'] = buildings.apply(get_energy_label, axis=1)
        
        # Insulation quality
        label_insulation = {
            'A': 'excellent', 'B': 'good', 'C': 'good',
            'D': 'fair', 'E': 'fair', 'F': 'poor', 'G': 'poor'
        }
        buildings['insulation_quality'] = buildings['energy_label'].map(label_insulation)
        
        # Solar potential calculation
        buildings['suitable_roof_area'] = buildings['flat_roof_area'] + \
            buildings.apply(lambda x: x['sloped_roof_area'] * 0.6 
                          if x['building_orientation_cardinal'] in ['S', 'SE', 'SW'] 
                          else x['sloped_roof_area'] * 0.3, axis=1)
        
        def get_solar_potential(row):
            if row['suitable_roof_area'] > 100:
                return 'high'
            elif row['suitable_roof_area'] > 50:
                return 'medium'
            elif row['suitable_roof_area'] > 25:
                return 'low'
            else:
                return 'none'
        
        buildings['solar_potential'] = buildings.apply(get_solar_potential, axis=1)
        buildings['solar_capacity_kwp'] = buildings['suitable_roof_area'] * 0.15 * 0.7
        
        # Battery readiness
        buildings['battery_readiness'] = buildings['solar_potential'].apply(
            lambda x: 'ready' if x in ['high', 'medium'] else 'conditional'
        )
        
        # Electrification feasibility
        def get_electrification_feasibility(row):
            if row['energy_label'] in ['F', 'G']:
                return 'upgrade_needed'
            elif row['energy_label'] in ['D', 'E']:
                return 'conditional'
            else:
                return 'immediate'
        
        buildings['electrification_feasibility'] = buildings.apply(get_electrification_feasibility, axis=1)
        
        # Expected COP for heat pumps
        cop_by_label = {
            'A': 4.0, 'B': 3.5, 'C': 3.0,
            'D': 2.5, 'E': 2.0, 'F': 1.8, 'G': 1.5
        }
        buildings['expected_cop'] = buildings['energy_label'].map(cop_by_label)
        
        # Current assets (simplified assumptions for demo)
        np.random.seed(42)  # For reproducibility
        buildings['has_solar'] = (buildings['solar_capacity_kwp'] > 5) & (np.random.random(len(buildings)) > 0.7)
        buildings['has_battery'] = buildings['has_solar'] & (np.random.random(len(buildings)) > 0.5)
        buildings['has_heat_pump'] = (buildings['energy_label'].isin(['A', 'B'])) & (np.random.random(len(buildings)) > 0.6)
        buildings['heating_system'] = buildings.apply(
            lambda x: 'heat_pump' if x['has_heat_pump'] else 'gas', axis=1
        )
        
        return buildings
    
    # ============================================
    # STEP 4: CREATE EXISTING ASSET NODES
    # ============================================
    
    def create_existing_assets(self):
        """Create asset nodes for buildings that already have installations"""
        logger.info("Creating existing asset nodes...")
        
        with self.driver.session() as session:
            # Existing solar systems
            result = session.run("""
                MATCH (b:Building {has_solar: true})
                CREATE (s:SolarSystem {
                    system_id: 'SOLAR_EXISTING_' + toString(b.ogc_fid),
                    building_id: b.ogc_fid,
                    status: 'existing',
                    installed_capacity_kwp: b.solar_capacity_kwp,
                    installation_year: 2023,
                    degradation_factor: 0.98,
                    orientation_efficiency: 
                        CASE b.building_orientation_cardinal
                            WHEN 'S' THEN 1.0
                            WHEN 'SE' THEN 0.95
                            WHEN 'SW' THEN 0.95
                            ELSE 0.8
                        END
                })
                CREATE (b)-[:HAS_INSTALLED {
                    install_date: date('2023-01-01')
                }]->(s)
                RETURN count(*) as count
            """)
            solar_count = result.single()['count']
            
            # Existing battery systems
            result = session.run("""
                MATCH (b:Building {has_battery: true})
                CREATE (bat:BatterySystem {
                    system_id: 'BATTERY_EXISTING_' + toString(b.ogc_fid),
                    building_id: b.ogc_fid,
                    status: 'existing',
                    installed_capacity_kwh: 
                        CASE 
                            WHEN b.area < 150 THEN 5.0
                            WHEN b.area < 300 THEN 10.0
                            ELSE 15.0
                        END,
                    power_rating_kw: 
                        CASE 
                            WHEN b.area < 150 THEN 2.5
                            WHEN b.area < 300 THEN 5.0
                            ELSE 7.5
                        END,
                    round_trip_efficiency: 0.9
                })
                CREATE (b)-[:HAS_INSTALLED {
                    install_date: date('2023-01-01')
                }]->(bat)
                RETURN count(*) as count
            """)
            battery_count = result.single()['count']
            
            # Existing heat pump systems
            result = session.run("""
                MATCH (b:Building {has_heat_pump: true})
                CREATE (hp:HeatPumpSystem {
                    system_id: 'HP_EXISTING_' + toString(b.ogc_fid),
                    building_id: b.ogc_fid,
                    status: 'existing',
                    expected_cop: b.expected_cop,
                    heating_capacity_kw: b.area * 0.05,
                    installation_year: 2022
                })
                CREATE (b)-[:HAS_INSTALLED {
                    install_date: date('2022-01-01')
                }]->(hp)
                RETURN count(*) as count
            """)
            hp_count = result.single()['count']
            
            self.stats['nodes_created']['ExistingSolar'] = solar_count
            self.stats['nodes_created']['ExistingBattery'] = battery_count
            self.stats['nodes_created']['ExistingHeatPump'] = hp_count
            self.stats['relationships_created']['HAS_INSTALLED'] = solar_count + battery_count + hp_count
            
        logger.info(f"Created existing assets: {solar_count} solar, {battery_count} battery, {hp_count} heat pumps")
    
    # ============================================
    # STEP 5: LOAD TEMPORAL ENERGY DATA
    # ============================================
    
    def load_temporal_data(self, data_path: str):
        """Load time-series energy profiles"""
        logger.info("Loading temporal energy data...")
        
        # Load energy profiles
        profiles = pd.read_parquet(f"{data_path}/energy_profiles.parquet")
        
        # Get unique timestamps
        timestamps = profiles['timestamp'].unique()
        
        with self.driver.session() as session:
            # Create TimeSlot nodes
            for i, ts in enumerate(timestamps):
                dt = pd.to_datetime(ts)
                session.run("""
                    CREATE (t:TimeSlot {
                        slot_id: $slot_id,
                        timestamp: datetime($timestamp),
                        hour_of_day: $hour,
                        day_of_week: $dow,
                        is_weekend: $weekend,
                        season: $season,
                        time_of_day: $tod
                    })
                """,
                slot_id=f"TS_{i}",
                timestamp=dt.isoformat(),
                hour=dt.hour,
                dow=dt.dayofweek,
                weekend=dt.dayofweek >= 5,
                season=self._get_season(dt),
                tod=self._get_time_of_day(dt.hour)
                )
            
            self.stats['nodes_created']['TimeSlot'] = len(timestamps)
            
            # Create EnergyState nodes in batches
            logger.info("Creating energy states (this may take a moment)...")
            
            batch_size = 1000
            state_count = 0
            
            for building_id in profiles['building_id'].unique():
                building_data = profiles[profiles['building_id'] == building_id]
                
                states = []
                for idx, row in building_data.iterrows():
                    dt = pd.to_datetime(row['timestamp'])
                    slot_id = f"TS_{list(timestamps).index(row['timestamp'])}"
                    
                    # Calculate net position
                    net_demand = row['electricity_demand_kw'] - row['solar_generation_kw'] + row['battery_discharge_kw'] - row['battery_charge_kw']
                    
                    states.append({
                        'state_id': f"ES_{building_id}_{slot_id}",
                        'building_id': int(building_id),
                        'timeslot_id': slot_id,
                        'electricity_demand_kw': float(row['electricity_demand_kw']),
                        'heating_demand_kw': float(row['heating_demand_kw']),
                        'cooling_demand_kw': float(row['cooling_demand_kw']),
                        'solar_generation_kw': float(row['solar_generation_kw']),
                        'battery_soc_kwh': float(row['battery_soc_kwh']),
                        'battery_charge_kw': float(row['battery_charge_kw']),
                        'battery_discharge_kw': float(row['battery_discharge_kw']),
                        'net_demand_kw': float(net_demand),
                        'is_surplus': net_demand < 0,
                        'export_potential_kw': max(0, -net_demand),
                        'import_need_kw': max(0, net_demand)
                    })
                    
                    if len(states) >= batch_size:
                        self._create_energy_states_batch(session, states)
                        state_count += len(states)
                        states = []
                
                if states:
                    self._create_energy_states_batch(session, states)
                    state_count += len(states)
            
            self.stats['nodes_created']['EnergyState'] = state_count
            
            # Create relationships
            logger.info("Creating temporal relationships...")
            
            # Building -> EnergyState
            result = session.run("""
                MATCH (b:Building), (e:EnergyState {building_id: b.ogc_fid})
                CREATE (b)-[:HAS_STATE_AT]->(e)
                RETURN count(*) as count
            """)
            self.stats['relationships_created']['HAS_STATE_AT'] = result.single()['count']
            
            # EnergyState -> TimeSlot
            result = session.run("""
                MATCH (e:EnergyState), (t:TimeSlot {slot_id: e.timeslot_id})
                CREATE (e)-[:DURING]->(t)
                RETURN count(*) as count
            """)
            self.stats['relationships_created']['DURING'] = result.single()['count']
            
        logger.info(f"Created {state_count} energy states with temporal relationships")
    
    def _create_energy_states_batch(self, session, states):
        """Create energy state nodes in batch"""
        session.run("""
            UNWIND $states as state
            CREATE (e:EnergyState {
                state_id: state.state_id,
                building_id: state.building_id,
                timeslot_id: state.timeslot_id,
                electricity_demand_kw: state.electricity_demand_kw,
                heating_demand_kw: state.heating_demand_kw,
                cooling_demand_kw: state.cooling_demand_kw,
                solar_generation_kw: state.solar_generation_kw,
                battery_soc_kwh: state.battery_soc_kwh,
                battery_charge_kw: state.battery_charge_kw,
                battery_discharge_kw: state.battery_discharge_kw,
                net_demand_kw: state.net_demand_kw,
                is_surplus: state.is_surplus,
                export_potential_kw: state.export_potential_kw,
                import_need_kw: state.import_need_kw
            })
        """, states=states)
    
    def _get_season(self, dt):
        """Get season from datetime"""
        month = dt.month
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'fall'
    
    def _get_time_of_day(self, hour):
        """Categorize time of day"""
        if 6 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 18:
            return 'afternoon'
        elif 18 <= hour < 22:
            return 'evening'
        else:
            return 'night'
    
    # ============================================
    # STEP 6: IDENTIFY ASSET OPPORTUNITIES
    # ============================================
    
    def identify_asset_opportunities(self):
        """Create nodes for potential solar, battery, and electrification"""
        logger.info("Identifying asset deployment opportunities...")
        
        with self.driver.session() as session:
            # Solar opportunities (for buildings without solar)
            result = session.run("""
                MATCH (b:Building)
                WHERE b.solar_potential IN ['high', 'medium']
                  AND b.has_solar = false
                CREATE (s:SolarSystem {
                    system_id: 'SOLAR_POTENTIAL_' + toString(b.ogc_fid),
                    building_id: b.ogc_fid,
                    status: 'potential',
                    potential_capacity_kwp: b.solar_capacity_kwp,
                    recommended_capacity_kwp: 
                        CASE 
                            WHEN b.area < 150 THEN 4.0
                            WHEN b.area < 300 THEN 6.0
                            ELSE 10.0
                        END,
                    orientation_efficiency: 
                        CASE b.building_orientation_cardinal
                            WHEN 'S' THEN 1.0
                            WHEN 'SE' THEN 0.95
                            WHEN 'SW' THEN 0.95
                            WHEN 'E' THEN 0.85
                            WHEN 'W' THEN 0.85
                            ELSE 0.7
                        END,
                    payback_years: 7.5
                })
                CREATE (b)-[:CAN_INSTALL {
                    feasibility_score: 
                        CASE b.solar_potential
                            WHEN 'high' THEN 0.9
                            WHEN 'medium' THEN 0.7
                            ELSE 0.5
                        END,
                    priority: 'medium'
                }]->(s)
                RETURN count(*) as count
            """)
            
            solar_count = result.single()['count']
            self.stats['nodes_created']['SolarPotential'] = solar_count
            
            # Battery opportunities
            result = session.run("""
                MATCH (b:Building)
                WHERE (b.solar_potential IN ['high', 'medium'] OR b.building_function = 'non_residential')
                  AND b.has_battery = false
                WITH b, 
                     CASE 
                         WHEN b.area < 150 THEN 5.0
                         WHEN b.area < 300 THEN 10.0
                         ELSE 20.0
                     END as battery_size
                CREATE (bat:BatterySystem {
                    system_id: 'BATTERY_POTENTIAL_' + toString(b.ogc_fid),
                    building_id: b.ogc_fid,
                    status: 'potential',
                    recommended_capacity_kwh: battery_size,
                    power_rating_kw: battery_size / 4,
                    round_trip_efficiency: 0.9,
                    estimated_cycles_per_year: 250
                })
                CREATE (b)-[:CAN_INSTALL {
                    feasibility_score: 
                        CASE 
                            WHEN b.has_solar OR b.solar_potential IN ['high', 'medium'] THEN 0.8
                            ELSE 0.6
                        END,
                    requires_solar: NOT b.has_solar,
                    priority: 'low'
                }]->(bat)
                RETURN count(*) as count
            """)
            
            battery_count = result.single()['count']
            self.stats['nodes_created']['BatteryPotential'] = battery_count
            
            # Electrification opportunities (heat pumps for poor efficiency buildings)
            result = session.run("""
                MATCH (b:Building)
                WHERE b.energy_label IN ['D', 'E', 'F', 'G']
                  AND b.has_heat_pump = false
                CREATE (hp:HeatPumpSystem {
                    system_id: 'HP_POTENTIAL_' + toString(b.ogc_fid),
                    building_id: b.ogc_fid,
                    status: 'potential',
                    expected_cop: b.expected_cop,
                    heating_capacity_kw: b.area * 0.05,
                    upgrade_required: b.electrification_feasibility = 'upgrade_needed',
                    estimated_annual_savings_euro: 
                        CASE b.energy_label
                            WHEN 'G' THEN 2000
                            WHEN 'F' THEN 1500
                            WHEN 'E' THEN 1000
                            ELSE 500
                        END
                })
                CREATE (b)-[:SHOULD_ELECTRIFY {
                    priority: 
                        CASE b.energy_label
                            WHEN 'G' THEN 1
                            WHEN 'F' THEN 2
                            WHEN 'E' THEN 3
                            WHEN 'D' THEN 4
                            ELSE 5
                        END,
                    expected_cop: b.expected_cop,
                    requires_insulation_upgrade: b.energy_label IN ['F', 'G']
                }]->(hp)
                RETURN count(*) as count
            """)
            
            hp_count = result.single()['count']
            self.stats['nodes_created']['HeatPumpPotential'] = hp_count
            
        logger.info(f"Identified opportunities: {solar_count} solar, {battery_count} battery, {hp_count} heat pumps")
    
    # ============================================
    # STEP 7: CALCULATE BASELINE METRICS
    # ============================================
    
    def calculate_baseline_metrics(self):
        """Calculate current state metrics for comparison after GNN optimization"""
        logger.info("Calculating baseline metrics...")
        
        with self.driver.session() as session:
            # Building-level statistics
            session.run("""
                MATCH (b:Building)-[:HAS_STATE_AT]->(e:EnergyState)
                WITH b, 
                    max(e.electricity_demand_kw) as peak_demand,
                    avg(e.electricity_demand_kw) as avg_demand,
                    stdev(e.electricity_demand_kw) as demand_std,
                    max(e.solar_generation_kw) as peak_solar,
                    avg(e.net_demand_kw) as avg_net_demand
                SET b.peak_demand_kw = peak_demand,
                    b.avg_demand_kw = avg_demand,
                    b.load_factor = avg_demand / CASE WHEN peak_demand > 0 THEN peak_demand ELSE 1 END,
                    b.demand_variability = demand_std / CASE WHEN avg_demand > 0 THEN avg_demand ELSE 1 END,
                    b.peak_solar_kw = peak_solar,
                    b.avg_net_demand_kw = avg_net_demand,
                    b.self_consumption_ratio = 
                        CASE WHEN b.has_solar 
                        THEN (avg_demand - avg_net_demand) / CASE WHEN avg_demand > 0 THEN avg_demand ELSE 1 END
                        ELSE 0 END
            """)
            
            # LV Network baseline statistics
            session.run("""
                MATCH (l:LV_Network)<-[:CONNECTED_TO]-(b:Building)
                WITH l, 
                    count(b) as building_count,
                    sum(b.peak_demand_kw) as total_peak_demand,
                    avg(b.peak_demand_kw) as avg_building_peak,
                    sum(b.avg_demand_kw) as total_avg_demand,
                    sum(CASE WHEN b.has_solar THEN 1 ELSE 0 END) as solar_count,
                    sum(CASE WHEN b.has_battery THEN 1 ELSE 0 END) as battery_count,
                    sum(CASE WHEN b.has_heat_pump THEN 1 ELSE 0 END) as hp_count,
                    collect(DISTINCT b.building_function) as building_types
                SET l.baseline_building_count = building_count,
                    l.baseline_peak_kw = total_peak_demand,
                    l.baseline_avg_demand_kw = total_avg_demand,
                    l.baseline_load_factor = total_avg_demand / CASE WHEN total_peak_demand > 0 THEN total_peak_demand ELSE 1 END,
                    l.baseline_solar_penetration = toFloat(solar_count) / building_count,
                    l.baseline_battery_penetration = toFloat(battery_count) / building_count,
                    l.baseline_hp_penetration = toFloat(hp_count) / building_count,
                    l.baseline_diversity = size(building_types),
                    l.baseline_self_sufficiency = 0.1  // Placeholder - will be calculated properly
            """)
            
            # Calculate peak coincidence factor - FIXED VERSION
            session.run("""
                MATCH (l:LV_Network)<-[:CONNECTED_TO]-(b:Building)
                WITH l, sum(b.peak_demand_kw) as sum_of_peaks
                MATCH (l)<-[:CONNECTED_TO]-(b2:Building)-[:HAS_STATE_AT]->(e:EnergyState)-[:DURING]->(t:TimeSlot)
                WITH l, sum_of_peaks, t, sum(e.electricity_demand_kw) as total_demand_at_time
                WITH l, sum_of_peaks, max(total_demand_at_time) as actual_peak
                SET l.baseline_coincidence_factor = actual_peak / CASE WHEN sum_of_peaks > 0 THEN sum_of_peaks ELSE 1 END
            """)
            
            # MV Transformer baseline
            session.run("""
                MATCH (m:MV_Transformer)<-[:FEEDS_FROM]-(l:LV_Network)
                WITH m, 
                    count(l) as lv_count,
                    sum(l.baseline_peak_kw) as total_peak,
                    sum(l.baseline_building_count) as total_buildings
                SET m.baseline_lv_count = lv_count,
                    m.baseline_peak_kw = total_peak,
                    m.baseline_utilization = total_peak / m.capacity_kva,
                    m.baseline_building_count = total_buildings
            """)
            
            # System-wide baseline
            result = session.run("""
                MATCH (b:Building)
                WITH count(b) as total_buildings,
                    sum(b.peak_demand_kw) as system_peak,
                    avg(b.load_factor) as avg_load_factor,
                    sum(CASE WHEN b.has_solar THEN 1 ELSE 0 END) as solar_buildings,
                    sum(CASE WHEN b.has_battery THEN 1 ELSE 0 END) as battery_buildings,
                    sum(CASE WHEN b.has_heat_pump THEN 1 ELSE 0 END) as hp_buildings
                CREATE (s:SystemBaseline {
                    id: 'BASELINE_' + toString(datetime()),
                    timestamp: datetime(),
                    total_buildings: total_buildings,
                    system_peak_kw: system_peak,
                    avg_load_factor: avg_load_factor,
                    solar_penetration: toFloat(solar_buildings) / total_buildings,
                    battery_penetration: toFloat(battery_buildings) / total_buildings,
                    hp_penetration: toFloat(hp_buildings) / total_buildings,
                    description: 'Pre-GNN optimization baseline'
                })
                RETURN total_buildings, system_peak, avg_load_factor
            """)
            
            baseline = result.single()
            logger.info(f"Baseline: {baseline['total_buildings']} buildings, "
                    f"{baseline['system_peak']:.0f} kW peak, "
                    f"{baseline['avg_load_factor']:.3f} load factor")
            
    # ============================================
    # STEP 8: ADD METADATA
    # ============================================
    
    def add_metadata(self):
        """Add metadata about the KG creation"""
        logger.info("Adding metadata...")
        
        with self.driver.session() as session:
            metadata = {
                'creation_timestamp': datetime.now().isoformat(),
                'total_nodes': sum(self.stats['nodes_created'].values()),
                'total_relationships': sum(self.stats['relationships_created'].values()),
                'node_types': self.stats['nodes_created'],
                'relationship_types': self.stats['relationships_created']
            }
            
            session.run("""
                CREATE (m:Metadata {
                    id: 'PRE_GNN_METADATA',
                    created_at: datetime($timestamp),
                    total_nodes: $nodes,
                    total_relationships: $rels,
                    node_breakdown: $node_types,
                    relationship_breakdown: $rel_types,
                    data_quality: 'mimic_data',
                    version: '1.0',
                    stage: 'pre_gnn',
                    description: 'Knowledge graph before GNN optimization'
                })
            """,
            timestamp=metadata['creation_timestamp'],
            nodes=metadata['total_nodes'],
            rels=metadata['total_relationships'],
            node_types=json.dumps(metadata['node_types']),
            rel_types=json.dumps(metadata['relationship_types'])
            )
            
        logger.info("Metadata added")
    
    # ============================================
    # MAIN EXECUTION METHOD
    # ============================================
    
    def build_complete_graph(self, data_path: str, clear_first: bool = True):
        """Build complete knowledge graph from mimic data (Pre-GNN version)"""
        
        start_time = datetime.now()
        logger.info("="*50)
        logger.info("Starting Knowledge Graph Construction (Pre-GNN)")
        logger.info("="*50)
        
        try:
            # Clear database if requested
            if clear_first:
                self.clear_database()
            
            # Build graph step by step
            self.create_schema()
            self.load_grid_infrastructure(data_path)
            self.load_buildings(data_path)
            self.create_existing_assets()
            self.load_temporal_data(data_path)
            self.identify_asset_opportunities()
            self.calculate_baseline_metrics()
            self.add_metadata()
            
            # Calculate total time
            end_time = datetime.now()
            total_time = (end_time - start_time).total_seconds()
            
            # Print summary
            logger.info("="*50)
            logger.info("Pre-GNN Knowledge Graph Construction Complete!")
            logger.info("="*50)
            logger.info(f"Total processing time: {total_time:.2f} seconds")
            logger.info("\nNodes created:")
            for node_type, count in self.stats['nodes_created'].items():
                logger.info(f"  {node_type}: {count:,}")
            logger.info(f"  TOTAL: {sum(self.stats['nodes_created'].values()):,}")
            
            logger.info("\nRelationships created:")
            for rel_type, count in self.stats['relationships_created'].items():
                logger.info(f"  {rel_type}: {count:,}")
            logger.info(f"  TOTAL: {sum(self.stats['relationships_created'].values()):,}")
            
            logger.info("\n⚠️ Note: Complementarity analysis and clustering will be added after GNN processing")
            
            return self.stats
            
        except Exception as e:
            logger.error(f"Error building knowledge graph: {str(e)}")
            raise

# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    # Configuration
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "aminasad"  # Change this to your password
    DATA_PATH = "mimic_data"
    
    # Create builder and construct graph
    builder = EnergyKnowledgeGraphBuilder(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    try:
        # Build complete knowledge graph
        stats = builder.build_complete_graph(DATA_PATH, clear_first=True)
        
        # Run validation queries
        with builder.driver.session() as session:
            # Check baseline metrics
            result = session.run("""
                MATCH (l:LV_Network)
                RETURN l.network_id as lv_network,
                       l.baseline_building_count as buildings,
                       l.baseline_peak_kw as peak_kw,
                       l.baseline_load_factor as load_factor,
                       l.baseline_solar_penetration as solar_pen
                ORDER BY buildings DESC
            """)
            
            print("\n" + "="*50)
            print("LV Network Baseline Summary:")
            print("="*50)
            for record in result:
                print(f"{record['lv_network']}: {record['buildings']} buildings, "
                      f"Peak: {record['peak_kw']:.0f} kW, "
                      f"LF: {record['load_factor']:.2f}, "
                      f"Solar: {record['solar_pen']:.1%}")
            
            # Check opportunities
            result = session.run("""
                MATCH (s:SolarSystem {status: 'potential'})
                WITH count(s) as solar_opp
                MATCH (b:BatterySystem {status: 'potential'})
                WITH solar_opp, count(b) as battery_opp
                MATCH (h:HeatPumpSystem {status: 'potential'})
                RETURN solar_opp, battery_opp, count(h) as hp_opp
            """)
            
            opp = result.single()
            print("\n" + "="*50)
            print("Deployment Opportunities Identified:")
            print("="*50)
            print(f"Solar: {opp['solar_opp']} buildings")
            print(f"Battery: {opp['battery_opp']} buildings")
            print(f"Heat Pump: {opp['hp_opp']} buildings")
        
    finally:
        builder.close()
    
    print("\n✅ Pre-GNN Knowledge Graph construction complete!")
    print("Ready for GNN processing to discover complementarity and optimal clustering")
    print("You can explore the graph in Neo4j Browser at http://localhost:7474")