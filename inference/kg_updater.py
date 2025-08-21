# inference/kg_updater.py
"""
Update Neo4j Knowledge Graph with GNN inference results
Handles bidirectional synchronization between GNN and KG
"""

from neo4j import GraphDatabase
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging
import json

logger = logging.getLogger(__name__)

class KGUpdater:
    """Update Knowledge Graph with inference results"""
    
    def __init__(self, 
                 uri: str,
                 user: str,
                 password: str,
                 database: str = "neo4j"):
        """
        Initialize KG updater
        
        Args:
            uri: Neo4j connection URI
            user: Username
            password: Password
            database: Database name
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        
        # Track update statistics
        self.stats = {
            'nodes_created': 0,
            'nodes_updated': 0,
            'relationships_created': 0,
            'relationships_updated': 0,
            'properties_updated': 0
        }
        
        logger.info(f"Connected to Neo4j at {uri}")
    
    def close(self):
        """Close database connection"""
        self.driver.close()
        logger.info("Neo4j connection closed")
    
    def update(self, results: Dict, task: str, metadata: Optional[Dict] = None) -> Dict:
        """
        Update KG with inference results
        
        Args:
            results: Inference results
            task: Task type
            metadata: Additional metadata
            
        Returns:
            Update statistics
        """
        logger.info(f"Updating KG with {task} results")
        
        # Reset statistics
        self.stats = {k: 0 for k in self.stats}
        
        # Route to appropriate update method
        if task == 'clustering':
            self._update_clustering(results.get('clustering', results))
        elif task == 'solar':
            self._update_solar(results.get('solar', results))
        elif task == 'retrofit':
            self._update_retrofit(results.get('retrofit', results))
        elif task == 'electrification':
            self._update_electrification(results.get('electrification', results))
        elif task == 'battery':
            self._update_battery(results.get('battery', results))
        elif task == 'p2p':
            self._update_p2p_trading(results.get('p2p', results))
        elif task == 'congestion':
            self._update_congestion(results.get('congestion', results))
        elif task == 'thermal':
            self._update_thermal_sharing(results.get('thermal', results))
        else:
            logger.warning(f"Unknown task: {task}")
        
        # Add metadata
        if metadata:
            self._add_metadata(task, metadata)
        
        logger.info(f"KG update complete: {self.stats}")
        
        return self.stats
    
    def _update_clustering(self, results: Dict):
        """Update clustering results"""
        with self.driver.session(database=self.database) as session:
            # Create cluster nodes
            if 'clusters' in results:
                for cluster_id, building_indices in results['clusters'].items():
                    # Create cluster node
                    cluster_props = {
                        'cluster_id': f"CLUSTER_{cluster_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        'num_buildings': len(building_indices),
                        'created_at': datetime.now().isoformat(),
                        'confidence': results.get('confidence', 0),
                        'modularity': results.get('modularity', 0)
                    }
                    
                    # Add metrics if available
                    if 'metrics' in results:
                        cluster_props.update({
                            'peak_reduction': results['metrics'].get('peak_reduction', 0),
                            'self_sufficiency': results['metrics'].get('self_sufficiency', 0),
                            'avg_complementarity': results['metrics'].get('avg_complementarity', 0)
                        })
                    
                    # Create cluster node
                    session.run("""
                        CREATE (c:EnergyCluster $props)
                        RETURN c
                    """, props=cluster_props)
                    
                    self.stats['nodes_created'] += 1
                    
                    # Link buildings to cluster
                    for building_idx in building_indices:
                        session.run("""
                            MATCH (b:Building {index: $idx})
                            MATCH (c:EnergyCluster {cluster_id: $cluster_id})
                            CREATE (b)-[:BELONGS_TO {
                                assigned_at: datetime(),
                                confidence: $confidence
                            }]->(c)
                        """, idx=building_idx, 
                            cluster_id=cluster_props['cluster_id'],
                            confidence=results.get('confidence', 0))
                        
                        self.stats['relationships_created'] += 1
            
            # Create complementarity relationships
            if 'complementarity_pairs' in results:
                for pair in results['complementarity_pairs']:
                    session.run("""
                        MATCH (b1:Building {index: $idx1})
                        MATCH (b2:Building {index: $idx2})
                        MERGE (b1)-[r:COMPLEMENTS]-(b2)
                        SET r.score = $score,
                            r.correlation = $correlation,
                            r.discovered_at = datetime(),
                            r.discovered_by = 'GNN'
                    """, idx1=pair['building1'],
                        idx2=pair['building2'],
                        score=pair.get('score', 0),
                        correlation=pair.get('correlation', 0))
                    
                    self.stats['relationships_created'] += 1
    
    def _update_solar(self, results: Dict):
        """Update solar optimization results"""
        with self.driver.session(database=self.database) as session:
            # Update building solar scores
            if 'ranking' in results:
                for rank, building_idx in enumerate(results['ranking']):
                    capacity = results.get('capacities', [0] * len(results['ranking']))[rank]
                    
                    session.run("""
                        MATCH (b:Building {index: $idx})
                        SET b.solar_rank = $rank,
                            b.solar_capacity_recommended = $capacity,
                            b.solar_score = $score,
                            b.solar_roi_years = $roi,
                            b.solar_analysis_date = datetime()
                    """, idx=building_idx,
                        rank=rank + 1,
                        capacity=capacity,
                        score=results.get('scores', [0] * len(results['ranking']))[rank],
                        roi=results.get('roi_years', [0] * len(results['ranking']))[rank])
                    
                    self.stats['nodes_updated'] += 1
            
            # Create solar deployment scenario
            if 'total_capacity' in results:
                session.run("""
                    CREATE (s:SolarScenario {
                        scenario_id: $id,
                        total_capacity_kwp: $capacity,
                        num_installations: $num,
                        avg_roi_years: $roi,
                        created_at: datetime(),
                        status: 'RECOMMENDED'
                    })
                """, id=f"SOLAR_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    capacity=results['total_capacity'],
                    num=results.get('viable_count', 0),
                    roi=results.get('avg_roi', 0))
                
                self.stats['nodes_created'] += 1
    
    def _update_retrofit(self, results: Dict):
        """Update retrofit targeting results"""
        with self.driver.session(database=self.database) as session:
            # Update building retrofit priorities
            if 'priority_ranking' in results:
                for rank, building_idx in enumerate(results['priority_ranking']):
                    session.run("""
                        MATCH (b:Building {index: $idx})
                        SET b.retrofit_priority = $rank,
                            b.retrofit_savings_potential = $savings,
                            b.retrofit_cost_estimate = $cost,
                            b.retrofit_analysis_date = datetime()
                    """, idx=building_idx,
                        rank=rank + 1,
                        savings=results.get('savings', [0] * len(results['priority_ranking']))[rank],
                        cost=results.get('costs', [0] * len(results['priority_ranking']))[rank])
                    
                    self.stats['nodes_updated'] += 1
            
            # Create retrofit program
            if 'total_investment' in results:
                session.run("""
                    CREATE (r:RetrofitProgram {
                        program_id: $id,
                        total_investment: $investment,
                        total_savings_potential: $savings,
                        avg_savings_percent: $avg_savings,
                        created_at: datetime()
                    })
                """, id=f"RETROFIT_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    investment=results['total_investment'],
                    savings=results.get('total_savings_potential', 0),
                    avg_savings=results.get('avg_savings', 0))
                
                self.stats['nodes_created'] += 1
    
    def _update_electrification(self, results: Dict):
        """Update electrification readiness results"""
        with self.driver.session(database=self.database) as session:
            # Update building electrification status
            if 'readiness_distribution' in results:
                # Update ready buildings
                if 'ready_buildings' in results:
                    for building_idx in results['ready_buildings']:
                        session.run("""
                            MATCH (b:Building {index: $idx})
                            SET b.electrification_status = 'READY',
                                b.hp_capacity_recommended = $capacity,
                                b.electrification_assessment_date = datetime()
                        """, idx=building_idx,
                            capacity=results.get('hp_capacities', {}).get(building_idx, 0))
                        
                        self.stats['nodes_updated'] += 1
    
    def _update_battery(self, results: Dict):
        """Update battery placement results"""
        with self.driver.session(database=self.database) as session:
            # Update building battery scores
            if 'ranking' in results:
                for rank, building_idx in enumerate(results['ranking']):
                    session.run("""
                        MATCH (b:Building {index: $idx})
                        SET b.battery_rank = $rank,
                            b.battery_capacity_recommended = $capacity,
                            b.battery_value_score = $score,
                            b.battery_analysis_date = datetime()
                    """, idx=building_idx,
                        rank=rank + 1,
                        capacity=results.get('capacities', [0] * len(results['ranking']))[rank],
                        score=results.get('value_scores', [0] * len(results['ranking']))[rank])
                    
                    self.stats['nodes_updated'] += 1
    
    def _update_p2p_trading(self, results: Dict):
        """Update P2P trading results"""
        with self.driver.session(database=self.database) as session:
            # Create trading relationships
            if 'trading_pairs' in results:
                for pair in results['trading_pairs']:
                    session.run("""
                        MATCH (b1:Building {index: $idx1})
                        MATCH (b2:Building {index: $idx2})
                        MERGE (b1)-[t:TRADES_ENERGY_WITH]-(b2)
                        SET t.compatibility = $compatibility,
                            t.avg_volume_kwh = $volume,
                            t.avg_price_eur = $price,
                            t.trading_value_eur = $value,
                            t.established_at = datetime()
                    """, idx1=pair['building1'],
                        idx2=pair['building2'],
                        compatibility=pair.get('compatibility', 0),
                        volume=pair.get('volume', 0),
                        price=pair.get('price', 0),
                        value=pair.get('value', 0))
                    
                    self.stats['relationships_created'] += 1
    
    def _update_congestion(self, results: Dict):
        """Update congestion prediction results"""
        with self.driver.session(database=self.database) as session:
            # Create congestion alert nodes
            if 'alerts' in results:
                for alert in results['alerts']:
                    session.run("""
                        CREATE (a:CongestionAlert {
                            alert_id: $id,
                            location: $location,
                            severity: $severity,
                            probability: $probability,
                            horizon: $horizon,
                            predicted_at: datetime(),
                            expected_time: $expected
                        })
                    """, id=f"ALERT_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        location=alert.get('location', 'unknown'),
                        severity=alert.get('severity', 'unknown'),
                        probability=alert.get('probability', 0),
                        horizon=alert.get('horizon', 'unknown'),
                        expected=alert.get('expected_time', 'unknown'))
                    
                    self.stats['nodes_created'] += 1
    
    def _update_thermal_sharing(self, results: Dict):
        """Update thermal sharing opportunities"""
        with self.driver.session(database=self.database) as session:
            # Create thermal sharing relationships
            if 'sharing_pairs' in results:
                for pair in results['sharing_pairs']:
                    session.run("""
                        MATCH (b1:Building {index: $idx1})
                        MATCH (b2:Building {index: $idx2})
                        MERGE (b1)-[s:SHARES_THERMAL_ENERGY]-(b2)
                        SET s.compatibility = $compatibility,
                            s.heat_transfer_potential = $potential,
                            s.shared_wall_length = $wall_length,
                            s.discovered_at = datetime()
                    """, idx1=pair['building1'],
                        idx2=pair['building2'],
                        compatibility=pair.get('compatibility', 0),
                        potential=pair.get('heat_transfer_potential', 0),
                        wall_length=pair.get('shared_wall_length', 0))
                    
                    self.stats['relationships_created'] += 1
    
    def _add_metadata(self, task: str, metadata: Dict):
        """Add metadata about the analysis"""
        with self.driver.session(database=self.database) as session:
            session.run("""
                CREATE (m:AnalysisMetadata {
                    analysis_id: $id,
                    task: $task,
                    timestamp: datetime(),
                    model_version: $model_version,
                    inference_time: $inference_time,
                    confidence: $confidence,
                    parameters: $parameters
                })
            """, id=f"{task.upper()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                task=task,
                model_version=metadata.get('model_version', 'unknown'),
                inference_time=metadata.get('inference_time', 0),
                confidence=metadata.get('confidence', 0),
                parameters=json.dumps(metadata.get('parameters', {})))
            
            self.stats['nodes_created'] += 1
    
    def bulk_update(self, results_list: List[Tuple[Dict, str]], 
                    batch_size: int = 1000) -> Dict:
        """
        Bulk update KG with multiple results
        
        Args:
            results_list: List of (results, task) tuples
            batch_size: Batch size for transactions
            
        Returns:
            Aggregated statistics
        """
        total_stats = {k: 0 for k in self.stats}
        
        with self.driver.session(database=self.database) as session:
            # Process in batches
            for i in range(0, len(results_list), batch_size):
                batch = results_list[i:i+batch_size]
                
                # Start transaction
                tx = session.begin_transaction()
                
                try:
                    for results, task in batch:
                        # Process each result
                        self.update(results, task)
                        
                        # Accumulate stats
                        for key, value in self.stats.items():
                            total_stats[key] += value
                    
                    # Commit transaction
                    tx.commit()
                    logger.info(f"Committed batch {i//batch_size + 1}")
                    
                except Exception as e:
                    tx.rollback()
                    logger.error(f"Batch update failed: {e}")
                    raise
        
        return total_stats
    
    def create_analysis_lineage(self, task: str, inputs: Dict, 
                               outputs: Dict) -> str:
        """
        Create lineage tracking for analysis
        
        Args:
            task: Task type
            inputs: Input parameters
            outputs: Output results
            
        Returns:
            Lineage ID
        """
        lineage_id = f"LINEAGE_{task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with self.driver.session(database=self.database) as session:
            # Create lineage node
            session.run("""
                CREATE (l:AnalysisLineage {
                    lineage_id: $id,
                    task: $task,
                    created_at: datetime(),
                    inputs: $inputs,
                    outputs: $outputs,
                    status: 'COMPLETED'
                })
            """, id=lineage_id,
                task=task,
                inputs=json.dumps(inputs),
                outputs=json.dumps(self._serialize_outputs(outputs)))
            
            # Link to affected nodes
            if 'affected_buildings' in outputs:
                for building_id in outputs['affected_buildings']:
                    session.run("""
                        MATCH (b:Building {id: $building_id})
                        MATCH (l:AnalysisLineage {lineage_id: $lineage_id})
                        CREATE (l)-[:AFFECTED]->(b)
                    """, building_id=building_id, lineage_id=lineage_id)
        
        return lineage_id
    
    def _serialize_outputs(self, outputs: Dict) -> Dict:
        """Serialize outputs for storage"""
        serialized = {}
        
        for key, value in outputs.items():
            if isinstance(value, np.ndarray):
                serialized[key] = value.tolist()
            elif isinstance(value, pd.DataFrame):
                serialized[key] = value.to_dict('records')
            elif isinstance(value, (list, dict, str, int, float, bool)):
                serialized[key] = value
            else:
                serialized[key] = str(value)
        
        return serialized
    
    def get_update_history(self, limit: int = 10) -> List[Dict]:
        """
        Get recent update history
        
        Args:
            limit: Number of records to retrieve
            
        Returns:
            List of update records
        """
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (m:AnalysisMetadata)
                RETURN m
                ORDER BY m.timestamp DESC
                LIMIT $limit
            """, limit=limit)
            
            history = []
            for record in result:
                metadata = dict(record['m'])
                history.append(metadata)
        
        return history

# Usage example
if __name__ == "__main__":
    import yaml
    
    # Load configuration
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize updater
    updater = KGUpdater(
        uri=config['neo4j']['uri'],
        user=config['neo4j']['user'],
        password=config['neo4j']['password']
    )
    
    # Mock results
    results = {
        'clustering': {
            'clusters': {
                0: [0, 1, 2, 3],
                1: [4, 5, 6, 7],
                2: [8, 9, 10]
            },
            'modularity': 0.65,
            'confidence': 0.85
        },
        'solar': {
            'ranking': [2, 5, 8, 1, 0],
            'capacities': [50, 45, 40, 35, 30],
            'total_capacity': 200,
            'avg_roi': 7.5
        }
    }
    
    # Update KG
    stats = updater.update(results, 'clustering')
    print(f"Clustering update: {stats}")
    
    stats = updater.update(results, 'solar')
    print(f"Solar update: {stats}")
    
    # Get history
    history = updater.get_update_history(5)
    print(f"Recent updates: {len(history)} records")
    
    # Close connection
    updater.close()