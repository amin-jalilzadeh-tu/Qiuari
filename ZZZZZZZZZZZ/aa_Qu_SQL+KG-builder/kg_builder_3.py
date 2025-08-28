# kg_builder_3_energy_data.py
"""
KG Builder 3: Add Energy Time Series Data from Parquet Files
Adds UBEM simulation results to the Knowledge Graph
"""

import pandas as pd
import numpy as np
from neo4j import GraphDatabase
from datetime import datetime
import pyarrow.parquet as pq
import logging
from tqdm import tqdm
from typing import Dict, List, Any, Optional
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnergyDataKGBuilder:
    """Builds KG with energy time series data from UBEM simulations."""
    
    def __init__(self, uri: str, user: str, password: str):
        """
        Initialize Neo4j connection.
        
        Args:
            uri: Neo4j connection URI
            user: Username
            password: Password
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info(f"Connected to Neo4j at {uri}")
        
        # Energy metrics to process
        self.energy_metrics = [
            'electricity_demand_kw',
            'heating_demand_kw', 
            'cooling_demand_kw',
            'solar_generation_kw',
            'battery_soc_kwh',
            'battery_charge_kw',
            'battery_discharge_kw'
        ]
        
    def load_parquet_data(self, file_path: str) -> pd.DataFrame:
        """
        Load energy data from parquet file.
        
        Args:
            file_path: Path to parquet file
            
        Returns:
            DataFrame with energy data
        """
        logger.info(f"Loading parquet file: {file_path}")
        
        # Read parquet file
        df = pd.read_parquet(file_path)
        
        # Convert timestamp to datetime if needed
        if 'timestamp' in df.columns:
            # If timestamp is in milliseconds (as shown in your example)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        logger.info(f"Loaded {len(df)} records for {df['building_id'].nunique()} buildings")
        logger.info(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        return df
    
    def create_time_slots(self, df: pd.DataFrame, batch_size: int = 1000):
        """
        Create TimeSlot nodes for unique timestamps.
        
        Args:
            df: DataFrame with energy data
            batch_size: Batch size for Neo4j transactions
        """
        logger.info("Creating TimeSlot nodes...")
        
        # Get unique timestamps
        unique_timestamps = df['timestamp'].unique()
        
        with self.driver.session() as session:
            # First create constraint for uniqueness
            try:
                session.run("""
                    CREATE CONSTRAINT timeslot_unique IF NOT EXISTS
                    FOR (ts:TimeSlot) REQUIRE ts.timeslot_id IS UNIQUE
                """)
            except:
                pass  # Constraint might already exist
            
            # Batch create TimeSlot nodes
            for i in tqdm(range(0, len(unique_timestamps), batch_size), desc="Creating TimeSlots"):
                batch = unique_timestamps[i:i+batch_size]
                
                timeslot_data = []
                for ts in batch:
                    ts_datetime = pd.Timestamp(ts)
                    timeslot_data.append({
                        'timeslot_id': ts_datetime.strftime('%Y%m%d_%H%M'),
                        'timestamp': int(ts_datetime.timestamp() * 1000),
                        'year': ts_datetime.year,
                        'month': ts_datetime.month,
                        'day': ts_datetime.day,
                        'hour': ts_datetime.hour,
                        'minute': ts_datetime.minute,
                        'day_of_week': ts_datetime.dayofweek,
                        'is_weekend': ts_datetime.dayofweek >= 5,
                        'season': self._get_season(ts_datetime.month)
                    })
                
                # Create nodes
                session.run("""
                    UNWIND $timeslots as ts
                    MERGE (t:TimeSlot {timeslot_id: ts.timeslot_id})
                    SET t += ts
                """, timeslots=timeslot_data)
        
        logger.info(f"Created {len(unique_timestamps)} TimeSlot nodes")
    
    def create_energy_states(self, df: pd.DataFrame, batch_size: int = 500):
        """
        Create EnergyState nodes linked to Buildings and TimeSlots.
        
        Args:
            df: DataFrame with energy data
            batch_size: Batch size for Neo4j transactions
        """
        logger.info("Creating EnergyState nodes...")
        
        # Ensure building_id column is consistently string type
        df['building_id_str'] = df['building_id'].astype(str)
        
        with self.driver.session() as session:
            # First verify buildings exist - with consistent string conversion
            building_ids_unique = df['building_id_str'].unique()
            
            logger.info(f"Checking existence of {len(building_ids_unique)} unique buildings...")
            
            # Query to find existing buildings
            result = session.run("""
                MATCH (b:Building)
                WHERE toString(b.ogc_fid) IN $ids
                RETURN toString(b.ogc_fid) as id
            """, ids=list(building_ids_unique))
            
            existing_buildings = {r['id'] for r in result}
            logger.info(f"Found {len(existing_buildings)} existing buildings out of {len(building_ids_unique)}")
            
            # Log some missing buildings for debugging
            missing = set(building_ids_unique) - existing_buildings
            if missing:
                sample_missing = list(missing)[:5]
                logger.warning(f"Sample missing building IDs: {sample_missing}")
            
            # Filter to only existing buildings
            df_filtered = df[df['building_id_str'].isin(existing_buildings)]
            
            if df_filtered.empty:
                logger.warning("No matching buildings found in KG!")
                return
            
            logger.info(f"Processing {len(df_filtered)} energy state records for {len(df_filtered['building_id_str'].unique())} buildings")
            
            # Create EnergyState nodes in batches
            total_records = len(df_filtered)
            created_count = 0
            error_count = 0
            
            for i in tqdm(range(0, total_records, batch_size), desc="Creating EnergyStates"):
                batch_df = df_filtered.iloc[i:i+batch_size]
                
                energy_states = []
                for _, row in batch_df.iterrows():
                    try:
                        # Ensure timestamp is properly handled
                        ts_datetime = pd.Timestamp(row['timestamp'])
                        
                        # Calculate net demand and export potential with null safety
                        electricity_demand = float(row.get('electricity_demand_kw', 0))
                        solar_generation = float(row.get('solar_generation_kw', 0))
                        battery_discharge = float(row.get('battery_discharge_kw', 0))
                        battery_charge = float(row.get('battery_charge_kw', 0))
                        
                        net_demand = (electricity_demand - 
                                    solar_generation - 
                                    battery_discharge + 
                                    battery_charge)
                        
                        export_potential = max(0, -net_demand)
                        
                        # Create state with consistent string IDs
                        state_id = f"{row['building_id_str']}_{ts_datetime.strftime('%Y%m%d_%H%M')}"
                        timeslot_id = ts_datetime.strftime('%Y%m%d_%H%M')
                        
                        energy_states.append({
                            'state_id': state_id,
                            'building_id': row['building_id_str'],  # Use string version
                            'timeslot_id': timeslot_id,
                            'electricity_demand_kw': electricity_demand,
                            'heating_demand_kw': float(row.get('heating_demand_kw', 0)),
                            'cooling_demand_kw': float(row.get('cooling_demand_kw', 0)),
                            'solar_generation_kw': solar_generation,
                            'battery_soc_kwh': float(row.get('battery_soc_kwh', 0)),
                            'battery_charge_kw': battery_charge,
                            'battery_discharge_kw': battery_discharge,
                            'net_demand_kw': float(net_demand),
                            'export_potential_kw': float(export_potential),
                            'total_demand_kw': float(
                                electricity_demand + 
                                row.get('heating_demand_kw', 0) + 
                                row.get('cooling_demand_kw', 0)
                            )
                        })
                    except Exception as e:
                        logger.debug(f"Error processing row: {e}")
                        error_count += 1
                        continue
                
                # Create nodes and relationships with better error handling
                if energy_states:
                    try:
                        # Updated query with consistent string handling
                        result = session.run("""
                            UNWIND $states as es
                            MATCH (b:Building) WHERE toString(b.ogc_fid) = es.building_id
                            MATCH (ts:TimeSlot {timeslot_id: es.timeslot_id})
                            MERGE (e:EnergyState {state_id: es.state_id})
                            SET e.building_id = es.building_id,
                                e.timeslot_id = es.timeslot_id,
                                e.electricity_demand_kw = es.electricity_demand_kw,
                                e.heating_demand_kw = es.heating_demand_kw,
                                e.cooling_demand_kw = es.cooling_demand_kw,
                                e.solar_generation_kw = es.solar_generation_kw,
                                e.battery_soc_kwh = es.battery_soc_kwh,
                                e.battery_charge_kw = es.battery_charge_kw,
                                e.battery_discharge_kw = es.battery_discharge_kw,
                                e.net_demand_kw = es.net_demand_kw,
                                e.export_potential_kw = es.export_potential_kw,
                                e.total_demand_kw = es.total_demand_kw
                            MERGE (e)-[:FOR_BUILDING]->(b)
                            MERGE (e)-[:DURING]->(ts)
                            RETURN count(e) as created
                        """, states=energy_states)
                        
                        # Get count of created nodes
                        record = result.single()
                        if record:
                            created_count += record['created']
                            
                    except Exception as e:
                        logger.error(f"Error creating batch of energy states: {e}")
                        error_count += len(energy_states)
        
        logger.info(f"Created {created_count} EnergyState nodes")
        if error_count > 0:
            logger.warning(f"Failed to create {error_count} energy states")
        
        # Verify creation
        with self.driver.session() as session:
            verify = session.run("""
                MATCH (es:EnergyState)
                RETURN count(es) as total_states
            """).single()
            
            if verify:
                logger.info(f"Total EnergyState nodes in database: {verify['total_states']}")
    
    def create_aggregated_profiles(self, df: pd.DataFrame):
        """
        Create aggregated energy profiles at different levels.
        """
        logger.info("Creating aggregated profiles...")
        
        with self.driver.session() as session:
            # Daily profiles per building - FIX ID HANDLING
            logger.info("Creating daily profiles...")
            session.run("""
                MATCH (es:EnergyState)-[:FOR_BUILDING]->(b:Building)
                MATCH (es)-[:DURING]->(ts:TimeSlot)
                WITH b, ts.year as year, ts.month as month, ts.day as day,
                    avg(es.electricity_demand_kw) as avg_elec,
                    max(es.electricity_demand_kw) as peak_elec,
                    sum(es.electricity_demand_kw) as total_elec,
                    avg(es.heating_demand_kw) as avg_heat,
                    max(es.heating_demand_kw) as peak_heat,
                    sum(es.heating_demand_kw) as total_heat,
                    avg(es.solar_generation_kw) as avg_solar,
                    max(es.solar_generation_kw) as peak_solar,
                    sum(es.solar_generation_kw) as total_solar
                MERGE (dp:DailyProfile {
                    profile_id: toString(b.ogc_fid) + '_' + toString(year) + 
                            toString(month) + toString(day)
                })
                SET dp.building_id = toString(b.ogc_fid),  // ENSURE STRING
                    dp.date = date({year: year, month: month, day: day}),
                    dp.avg_electricity_kw = avg_elec,
                    dp.peak_electricity_kw = peak_elec,
                    dp.total_electricity_kwh = total_elec,
                    dp.avg_heating_kw = avg_heat,
                    dp.peak_heating_kw = peak_heat,
                    dp.total_heating_kwh = total_heat,
                    dp.avg_solar_kw = avg_solar,
                    dp.peak_solar_kw = peak_solar,
                    dp.total_solar_kwh = total_solar
                MERGE (dp)-[:PROFILE_FOR]->(b)
            """)
            
            # Monthly profiles per building
            logger.info("Creating monthly profiles...")
            session.run("""
                MATCH (dp:DailyProfile)-[:PROFILE_FOR]->(b:Building)
                WITH b, dp.date.year as year, dp.date.month as month,
                     avg(dp.avg_electricity_kw) as avg_elec,
                     max(dp.peak_electricity_kw) as peak_elec,
                     sum(dp.total_electricity_kwh) as total_elec,
                     avg(dp.avg_heating_kw) as avg_heat,
                     max(dp.peak_heating_kw) as peak_heat,
                     sum(dp.total_heating_kwh) as total_heat
                MERGE (mp:MonthlyProfile {
                    profile_id: toString(b.ogc_fid) + '_' + toString(year) + toString(month)
                })
                SET mp.building_id = b.ogc_fid,
                    mp.year = year,
                    mp.month = month,
                    mp.avg_electricity_kw = avg_elec,
                    mp.peak_electricity_kw = peak_elec,
                    mp.total_electricity_kwh = total_elec,
                    mp.avg_heating_kw = avg_heat,
                    mp.peak_heating_kw = peak_heat,
                    mp.total_heating_kwh = total_heat
                MERGE (mp)-[:PROFILE_FOR]->(b)
            """)
            
            # Peak demand hours identification
            logger.info("Identifying peak demand hours...")
            session.run("""
                MATCH (es:EnergyState)-[:FOR_BUILDING]->(b:Building)
                MATCH (es)-[:DURING]->(ts:TimeSlot)
                WITH b, ts.hour as hour,
                     avg(es.electricity_demand_kw) as avg_demand,
                     count(*) as samples
                WHERE samples > 20  // Minimum samples for reliability
                WITH b, hour, avg_demand
                ORDER BY b.ogc_fid, avg_demand DESC
                WITH b, collect({hour: hour, demand: avg_demand})[0..3] as peak_hours
                SET b.peak_hours = [ph IN peak_hours | ph.hour],
                    b.peak_demands = [ph IN peak_hours | ph.demand]
            """)
    
    def create_cable_group_aggregates(self):
        """
        Aggregate energy data at cable group level.
        """
        logger.info("Creating cable group energy aggregates...")
        
        with self.driver.session() as session:
            session.run("""
                MATCH (cg:CableGroup)<-[:CONNECTED_TO]-(b:Building)
                OPTIONAL MATCH (b)<-[:PROFILE_FOR]-(mp:MonthlyProfile)
                WITH cg, 
                     avg(mp.avg_electricity_kw) as avg_group_elec,
                     max(mp.peak_electricity_kw) as peak_group_elec,
                     sum(mp.total_electricity_kwh) as total_group_elec,
                     avg(mp.avg_heating_kw) as avg_group_heat,
                     max(mp.peak_heating_kw) as peak_group_heat,
                     sum(mp.total_heating_kwh) as total_group_heat,
                     count(DISTINCT b) as building_count
                SET cg.avg_electricity_demand_kw = avg_group_elec,
                    cg.peak_electricity_demand_kw = peak_group_elec,
                    cg.total_electricity_demand_kwh = total_group_elec,
                    cg.avg_heating_demand_kw = avg_group_heat,
                    cg.peak_heating_demand_kw = peak_group_heat,
                    cg.total_heating_demand_kwh = total_group_heat,
                    cg.demand_diversity_factor = CASE 
                        WHEN building_count > 0 AND peak_group_elec > 0
                        THEN avg_group_elec * building_count / peak_group_elec
                        ELSE 1.0 END
            """)
    
    def create_adjacency_cluster_energy(self):
        """
        Calculate energy sharing potential for adjacency clusters.
        """
        logger.info("Calculating energy sharing potential...")
        
        with self.driver.session() as session:
            session.run("""
                MATCH (ac:AdjacencyCluster)<-[:IN_ADJACENCY_CLUSTER]-(b:Building)
                OPTIONAL MATCH (es:EnergyState)-[:FOR_BUILDING]->(b)
                WITH ac, b, 
                     avg(es.solar_generation_kw) as avg_solar,
                     avg(es.electricity_demand_kw) as avg_demand,
                     avg(es.export_potential_kw) as avg_export
                WITH ac,
                     sum(avg_solar) as cluster_solar_gen,
                     sum(avg_demand) as cluster_demand,
                     sum(avg_export) as cluster_export,
                     count(b) as building_count
                SET ac.total_solar_generation_kw = cluster_solar_gen,
                    ac.total_demand_kw = cluster_demand,
                    ac.export_potential_kw = cluster_export,
                    ac.self_sufficiency_ratio = CASE
                        WHEN cluster_demand > 0
                        THEN cluster_solar_gen / cluster_demand
                        ELSE 0 END,
                    ac.sharing_benefit_kwh = cluster_export * 0.8  // 80% efficiency
            """)
    
    def add_energy_features_to_buildings(self):
        """
        Add aggregated energy features directly to Building nodes.
        """
        logger.info("Adding energy features to buildings...")
        
        with self.driver.session() as session:
            session.run("""
                MATCH (b:Building)
                OPTIONAL MATCH (b)<-[:PROFILE_FOR]-(mp:MonthlyProfile)
                WITH b,
                     avg(mp.avg_electricity_kw) as avg_elec,
                     max(mp.peak_electricity_kw) as peak_elec,
                     avg(mp.avg_heating_kw) as avg_heat,
                     max(mp.peak_heating_kw) as peak_heat,
                     sum(mp.total_electricity_kwh) as annual_elec,
                     sum(mp.total_heating_kwh) as annual_heat
                SET b.avg_electricity_demand_kw = COALESCE(avg_elec, 0),
                    b.peak_electricity_demand_kw = COALESCE(peak_elec, 0),
                    b.avg_heating_demand_kw = COALESCE(avg_heat, 0),
                    b.peak_heating_demand_kw = COALESCE(peak_heat, 0),
                    b.annual_electricity_kwh = COALESCE(annual_elec, 0),
                    b.annual_heating_kwh = COALESCE(annual_heat, 0),
                    b.energy_intensity_kwh_m2 = CASE
                        WHEN b.area > 0
                        THEN (COALESCE(annual_elec, 0) + COALESCE(annual_heat, 0)) / b.area
                        ELSE 0 END
            """)
    
    def _get_season(self, month: int) -> str:
        """Get season from month."""
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'autumn'
    
    def create_indices(self):
        """Create indices for better query performance."""
        logger.info("Creating indices...")
        
        with self.driver.session() as session:
            indices = [
                "CREATE INDEX es_building IF NOT EXISTS FOR (es:EnergyState) ON (es.building_id)",
                "CREATE INDEX es_timeslot IF NOT EXISTS FOR (es:EnergyState) ON (es.timeslot_id)",
                "CREATE INDEX ts_timestamp IF NOT EXISTS FOR (ts:TimeSlot) ON (ts.timestamp)",
                "CREATE INDEX dp_building IF NOT EXISTS FOR (dp:DailyProfile) ON (dp.building_id)",
                "CREATE INDEX mp_building IF NOT EXISTS FOR (mp:MonthlyProfile) ON (mp.building_id)"
            ]
            
            for idx in indices:
                try:
                    session.run(idx)
                except:
                    pass  # Index might already exist
    
    def process_energy_data(self, parquet_path: str):
        """
        Main processing pipeline for energy data.
        
        Args:
            parquet_path: Path to parquet file with energy data
        """
        logger.info("="*50)
        logger.info("Starting Energy Data KG Builder")
        logger.info("="*50)
        
        # Load data
        df = self.load_parquet_data(parquet_path)
        
        # Create indices first
        self.create_indices()
        
        # Create time slots
        self.create_time_slots(df)
        
        # Create energy states
        self.create_energy_states(df)
        
        # Create aggregated profiles
        self.create_aggregated_profiles(df)
        
        # Aggregate at cable group level
        self.create_cable_group_aggregates()
        
        # Calculate cluster energy sharing
        self.create_adjacency_cluster_energy()
        
        # Add features to buildings
        self.add_energy_features_to_buildings()
        
        logger.info("="*50)
        logger.info("Energy Data KG Builder Complete!")
        logger.info("="*50)
        
        # Print summary statistics
        self._print_summary()
    
    def _print_summary(self):
        """Print summary statistics of created data."""
        with self.driver.session() as session:
            stats = session.run("""
                MATCH (es:EnergyState)
                WITH count(es) as energy_states
                MATCH (ts:TimeSlot)
                WITH energy_states, count(ts) as time_slots
                MATCH (dp:DailyProfile)
                WITH energy_states, time_slots, count(dp) as daily_profiles
                MATCH (mp:MonthlyProfile)
                WITH energy_states, time_slots, daily_profiles, count(mp) as monthly_profiles
                MATCH (b:Building) WHERE b.avg_electricity_demand_kw IS NOT NULL
                RETURN energy_states, time_slots, daily_profiles, monthly_profiles,
                       count(b) as buildings_with_energy
            """).single()
            
            if stats:
                logger.info("\n" + "="*50)
                logger.info("SUMMARY STATISTICS:")
                logger.info(f"  EnergyState nodes: {stats['energy_states']:,}")
                logger.info(f"  TimeSlot nodes: {stats['time_slots']:,}")
                logger.info(f"  DailyProfile nodes: {stats['daily_profiles']:,}")
                logger.info(f"  MonthlyProfile nodes: {stats['monthly_profiles']:,}")
                logger.info(f"  Buildings with energy data: {stats['buildings_with_energy']:,}")
                logger.info("="*50)
    
    def close(self):
        """Close Neo4j connection."""
        self.driver.close()
        logger.info("Neo4j connection closed")


# Main execution
if __name__ == "__main__":
    # Configuration
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "aminasad"
    
    # Path to your parquet file (with matched building IDs from KG)
    PARQUET_PATH = "mimic_data/energy_profiles_matched.parquet"
    
    # Create builder and process
    builder = EnergyDataKGBuilder(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    try:
        # Process the energy data
        builder.process_energy_data(PARQUET_PATH)
        
    finally:
        builder.close()