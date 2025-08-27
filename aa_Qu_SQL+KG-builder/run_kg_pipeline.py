"""
Run the complete KG building pipeline including optimization.
Execute builders 1, 2, 3, then apply optimizations with builder 4.
"""

import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_pipeline(skip_optimization=False):
    """
    Run the complete KG building pipeline.
    
    Args:
        skip_optimization: If True, skip the optimization step
    """
    # Import builders
    try:
        from kg_builder_1 import EnergyKnowledgeGraphBuilder
        from kg_builder_2 import GridInfrastructureKGBuilder  
        from kg_builder_3 import EnergyDataKGBuilder
        from kg_builder_4_optimizer import KGOptimizer
    except ImportError as e:
        logger.error(f"Failed to import builders: {e}")
        return
    
    # Connection parameters
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "password"  # Change this
    
    PG_HOST = "localhost"
    PG_DATABASE = "energy_db"
    PG_USER = "postgres"
    PG_PASSWORD = "password"  # Change this
    
    try:
        # Step 1: Build base KG
        logger.info("=" * 50)
        logger.info("STEP 1: Building base Knowledge Graph...")
        logger.info("=" * 50)
        
        builder1 = EnergyKnowledgeGraphBuilder(
            NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
            PG_HOST, PG_DATABASE, PG_USER, PG_PASSWORD
        )
        # builder1.build_knowledge_graph()  # Uncomment when ready to run
        logger.info("Base KG built successfully")
        
        # Step 2: Add grid infrastructure
        logger.info("=" * 50)
        logger.info("STEP 2: Adding grid infrastructure...")
        logger.info("=" * 50)
        
        builder2 = GridInfrastructureKGBuilder(
            NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
        )
        # builder2.build_infrastructure()  # Uncomment when ready to run
        logger.info("Grid infrastructure added successfully")
        
        # Step 3: Add energy data
        logger.info("=" * 50)
        logger.info("STEP 3: Adding energy time series data...")
        logger.info("=" * 50)
        
        builder3 = EnergyDataKGBuilder(
            NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
        )
        # builder3.add_energy_data()  # Uncomment when ready to run
        logger.info("Energy data added successfully")
        
        # Step 4: Optimize KG (NEW!)
        if not skip_optimization:
            logger.info("=" * 50)
            logger.info("STEP 4: Optimizing Knowledge Graph...")
            logger.info("=" * 50)
            
            optimizer = KGOptimizer(
                NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
            )
            optimizer.run_full_optimization()
            optimizer.close()
            logger.info("KG optimization completed successfully")
        else:
            logger.info("Skipping optimization step")
        
        logger.info("=" * 50)
        logger.info("KG PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    # Check if we should skip optimization
    skip_opt = "--skip-optimization" in sys.argv
    
    run_pipeline(skip_optimization=skip_opt)