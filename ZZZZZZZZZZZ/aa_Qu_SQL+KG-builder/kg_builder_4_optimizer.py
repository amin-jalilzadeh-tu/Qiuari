"""
KG Builder 4: Professional Optimization Layer
Run AFTER kg_builder_1, 2, and 3 to add:
- Indexes for query performance
- Constraints for data integrity
- Semantic properties (without ontology)
- Data validation
- Performance monitoring

This is a NON-DESTRUCTIVE optimizer that only adds optimizations.
"""

import logging
from neo4j import GraphDatabase
from datetime import datetime
import time
from typing import Dict, List, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KGOptimizer:
    """Optimize existing Knowledge Graph for professional performance."""
    
    def __init__(self, uri: str, user: str, password: str):
        """Initialize Neo4j connection."""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info(f"Connected to Neo4j for optimization")
        self.stats = {
            'indexes_created': [],
            'constraints_added': [],
            'semantic_properties': 0,
            'validation_issues': [],
            'performance_gains': {}
        }
    
    def create_indexes(self) -> None:
        """Create indexes for faster queries."""
        logger.info("Creating performance indexes...")
        
        indexes = [
            # Building indexes
            ("Building", "ogc_fid", "building_ogc_fid"),
            ("Building", "district_name", "building_district"),
            ("Building", "energy_label", "building_label"),
            ("Building", "building_function", "building_function"),
            
            # Infrastructure indexes
            ("CableGroup", "group_id", "cable_group_id"),
            ("CableGroup", "voltage_level", "cable_voltage"),
            ("Transformer", "ogc_fid", "transformer_id"),
            ("Substation", "name", "substation_name"),
            
            # Temporal indexes
            ("TimeSlot", "timestamp", "timeslot_timestamp"),
            ("TimeSlot", "hour", "timeslot_hour"),
            
            # Adjacency indexes
            ("AdjacencyCluster", "cluster_id", "adjacency_cluster_id")
        ]
        
        with self.driver.session() as session:
            for label, property_name, index_name in indexes:
                try:
                    # Check if index exists
                    check_query = "SHOW INDEXES YIELD name WHERE name = $index_name RETURN name"
                    result = session.run(check_query, index_name=index_name)
                    
                    if not result.single():
                        # Create index
                        query = f"CREATE INDEX {index_name} IF NOT EXISTS FOR (n:{label}) ON (n.{property_name})"
                        session.run(query)
                        self.stats['indexes_created'].append(index_name)
                        logger.info(f"Created index: {index_name}")
                    else:
                        logger.info(f"Index already exists: {index_name}")
                        
                except Exception as e:
                    logger.warning(f"Could not create index {index_name}: {e}")
    
    def add_constraints(self) -> None:
        """Add constraints for data integrity."""
        logger.info("Adding data integrity constraints...")
        
        constraints = [
            ("Building", "ogc_fid", "building_unique"),
            ("CableGroup", "group_id", "cable_group_unique"),
            ("Transformer", "ogc_fid", "transformer_unique"),
            ("Substation", "name", "substation_unique"),
            ("AdjacencyCluster", "cluster_id", "cluster_unique")
        ]
        
        with self.driver.session() as session:
            for label, property_name, constraint_name in constraints:
                try:
                    # Check if constraint exists
                    check_query = "SHOW CONSTRAINTS YIELD name WHERE name = $constraint_name RETURN name"
                    result = session.run(check_query, constraint_name=constraint_name)
                    
                    if not result.single():
                        # Create constraint
                        query = f"""CREATE CONSTRAINT {constraint_name} IF NOT EXISTS 
                                   FOR (n:{label}) REQUIRE n.{property_name} IS UNIQUE"""
                        session.run(query)
                        self.stats['constraints_added'].append(constraint_name)
                        logger.info(f"Created constraint: {constraint_name}")
                    else:
                        logger.info(f"Constraint already exists: {constraint_name}")
                        
                except Exception as e:
                    logger.warning(f"Could not create constraint {constraint_name}: {e}")
    
    def add_semantic_properties(self) -> None:
        """Add semantic metadata without full ontology."""
        logger.info("Adding semantic properties...")
        
        queries = [
            # Classify buildings by consumption
            """
            MATCH (b:Building)
            WHERE b.annual_consumption_kwh IS NOT NULL
            SET b.semantic_type = CASE
                WHEN b.annual_consumption_kwh > 50000 THEN 'HighConsumer'
                WHEN b.annual_consumption_kwh < 10000 THEN 'LowConsumer'
                ELSE 'NormalConsumer'
            END,
            b.semantic_category = 'EnergyAsset',
            b.semantic_validated = true
            RETURN count(b) as updated
            """,
            
            # Add efficiency classification
            """
            MATCH (b:Building)
            WHERE b.energy_label IS NOT NULL
            SET b.efficiency_class = CASE
                WHEN b.energy_label IN ['A', 'B', 'C'] THEN 'Efficient'
                WHEN b.energy_label IN ['D'] THEN 'Moderate'
                ELSE 'Inefficient'
            END
            RETURN count(b) as updated
            """,
            
            # Mark buildings with solar potential
            """
            MATCH (b:Building)
            WHERE b.solar_potential_kw IS NOT NULL
            SET b.renewable_ready = CASE
                WHEN b.solar_potential_kw > 10 THEN true
                ELSE false
            END
            RETURN count(b) as updated
            """,
            
            # Add grid position classification
            """
            MATCH (b:Building)-[:CONNECTED_TO]->(cg:CableGroup)
            SET b.grid_position = 'EndUser',
                cg.grid_position = 'Distribution'
            RETURN count(b) as updated
            """
        ]
        
        with self.driver.session() as session:
            for query in queries:
                result = session.run(query)
                count = result.single()['updated']
                self.stats['semantic_properties'] += count
                logger.info(f"Updated {count} nodes with semantic properties")
    
    def validate_data_quality(self) -> Dict[str, Any]:
        """Check data quality and report issues."""
        logger.info("Validating data quality...")
        
        validations = {
            'missing_properties': {},
            'invalid_values': {},
            'orphan_nodes': {},
            'integrity_issues': {}
        }
        
        with self.driver.session() as session:
            # Check for missing required properties
            missing_checks = [
                ("Building", ["ogc_fid", "district_name"]),
                ("CableGroup", ["group_id", "voltage_level"]),
                ("Transformer", ["ogc_fid"])
            ]
            
            for label, properties in missing_checks:
                for prop in properties:
                    query = f"""
                    MATCH (n:{label})
                    WHERE n.{prop} IS NULL
                    RETURN count(n) as count
                    """
                    result = session.run(query)
                    count = result.single()['count']
                    if count > 0:
                        validations['missing_properties'][f"{label}.{prop}"] = count
            
            # Check for invalid energy labels
            query = """
            MATCH (b:Building)
            WHERE b.energy_label IS NOT NULL 
            AND NOT b.energy_label IN ['A','B','C','D','E','F','G']
            RETURN count(b) as count
            """
            result = session.run(query)
            invalid_labels = result.single()['count']
            if invalid_labels > 0:
                validations['invalid_values']['energy_labels'] = invalid_labels
            
            # Check for orphan buildings (not connected to cable groups)
            query = """
            MATCH (b:Building)
            WHERE NOT EXISTS((b)-[:CONNECTED_TO]->(:CableGroup))
            RETURN count(b) as count
            """
            result = session.run(query)
            orphans = result.single()['count']
            if orphans > 0:
                validations['orphan_nodes']['unconnected_buildings'] = orphans
            
            self.stats['validation_issues'] = validations
            return validations
    
    def measure_performance(self) -> Dict[str, float]:
        """Measure query performance improvements."""
        logger.info("Measuring query performance...")
        
        test_queries = [
            ("Building lookup by ID", 
             "MATCH (b:Building {ogc_fid: 'FAA0000100'}) RETURN b"),
            
            ("District aggregation",
             "MATCH (b:Building {district_name: 'Hilversum Centrum'}) RETURN count(b)"),
            
            ("Time series query",
             "MATCH (es:EnergyState)-[:DURING]->(ts:TimeSlot) WHERE ts.hour = 12 RETURN count(es)"),
            
            ("Grid traversal",
             "MATCH path = (b:Building)-[:CONNECTED_TO]->(:CableGroup)-[:CONNECTS_TO]->(:Transformer) RETURN count(path)")
        ]
        
        performance = {}
        with self.driver.session() as session:
            for name, query in test_queries:
                start = time.time()
                session.run(query).consume()
                elapsed = time.time() - start
                performance[name] = elapsed
                logger.info(f"{name}: {elapsed:.3f}s")
        
        self.stats['performance_gains'] = performance
        return performance
    
    def create_composite_indexes(self) -> None:
        """Create composite indexes for complex queries."""
        logger.info("Creating composite indexes...")
        
        composite_indexes = [
            # Building composite indexes
            ("Building", ["district_name", "energy_label"], "building_district_label"),
            ("Building", ["building_function", "energy_label"], "building_function_label"),
            
            # Cable group composite
            ("CableGroup", ["voltage_level", "group_id"], "cable_voltage_group")
        ]
        
        with self.driver.session() as session:
            for label, properties, index_name in composite_indexes:
                try:
                    props_str = ", ".join([f"n.{p}" for p in properties])
                    query = f"CREATE INDEX {index_name} IF NOT EXISTS FOR (n:{label}) ON ({props_str})"
                    session.run(query)
                    self.stats['indexes_created'].append(index_name)
                    logger.info(f"Created composite index: {index_name}")
                except Exception as e:
                    logger.warning(f"Could not create composite index {index_name}: {e}")
    
    def optimize_database(self) -> None:
        """Run database optimization commands."""
        logger.info("Running database optimization...")
        
        with self.driver.session() as session:
            try:
                # Update database statistics
                session.run("CALL db.stats.retrieve('GRAPH')")
                logger.info("Updated database statistics")
                
                # Checkpoint to flush changes
                session.run("CALL db.checkpoint()")
                logger.info("Database checkpoint completed")
                
            except Exception as e:
                logger.warning(f"Optimization commands may not be available: {e}")
    
    def generate_report(self) -> str:
        """Generate optimization report."""
        report = f"""
========================================
KG OPTIMIZATION REPORT
Generated: {datetime.now()}
========================================

INDEXES CREATED: {len(self.stats['indexes_created'])}
{chr(10).join('  - ' + idx for idx in self.stats['indexes_created'])}

CONSTRAINTS ADDED: {len(self.stats['constraints_added'])}
{chr(10).join('  - ' + const for const in self.stats['constraints_added'])}

SEMANTIC PROPERTIES: {self.stats['semantic_properties']} nodes updated

DATA QUALITY VALIDATION:
"""
        
        for category, issues in self.stats['validation_issues'].items():
            if issues:
                report += f"\n  {category.upper()}:\n"
                for issue, count in issues.items():
                    report += f"    - {issue}: {count}\n"
        
        if self.stats['performance_gains']:
            report += "\nQUERY PERFORMANCE:\n"
            for query, time in self.stats['performance_gains'].items():
                report += f"  - {query}: {time:.3f}s\n"
        
        report += "\n========================================"
        return report
    
    def run_full_optimization(self) -> None:
        """Run all optimization steps."""
        logger.info("Starting full KG optimization...")
        start_time = time.time()
        
        try:
            # Step 1: Create indexes
            self.create_indexes()
            
            # Step 2: Add constraints
            self.add_constraints()
            
            # Step 3: Create composite indexes
            self.create_composite_indexes()
            
            # Step 4: Add semantic properties
            self.add_semantic_properties()
            
            # Step 5: Validate data
            self.validate_data_quality()
            
            # Step 6: Measure performance
            self.measure_performance()
            
            # Step 7: Optimize database
            self.optimize_database()
            
            elapsed = time.time() - start_time
            logger.info(f"Optimization completed in {elapsed:.2f} seconds")
            
            # Generate and print report
            report = self.generate_report()
            print(report)
            
            # Save report to file
            with open('kg_optimization_report.txt', 'w') as f:
                f.write(report)
            logger.info("Report saved to kg_optimization_report.txt")
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise
    
    def close(self):
        """Close database connection."""
        self.driver.close()


def main():
    """Run KG optimization."""
    # Neo4j connection parameters
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "password"  # Change this to your password
    
    # Create optimizer
    optimizer = KGOptimizer(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    try:
        # Run full optimization
        optimizer.run_full_optimization()
        
    finally:
        optimizer.close()


if __name__ == "__main__":
    main()