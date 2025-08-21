"""
Enhanced main.py section that integrates KG builders
This can be added to main.py or used as a separate utility
"""

import os
import sys
import yaml
import logging
from pathlib import Path
import pandas as pd
from typing import Dict, Optional

# Import the KG builders
from kg_builder_1 import EnergyKnowledgeGraphBuilder
from kg_builder_2 import AdjacencyUpdater

logger = logging.getLogger(__name__)

class KGManager:
    """Manages Knowledge Graph building and updates"""
    
    def __init__(self, config: Dict):
        """Initialize KG Manager with config"""
        self.config = config
        self.neo4j_config = {
            'uri': config.get('neo4j', {}).get('uri', 'bolt://localhost:7687'),
            'user': config.get('neo4j', {}).get('user', 'neo4j'),
            'password': config.get('neo4j', {}).get('password', 'password')
        }
        self.data_path = Path(config.get('paths', {}).get('raw_data', 'mimic_data'))
        
    def check_data_availability(self) -> Dict[str, bool]:
        """Check which data files are available"""
        required_files = {
            'buildings': self.data_path / 'buildings.csv',
            'lv_networks': self.data_path / 'lv_networks.csv',
            'mv_transformers': self.data_path / 'mv_transformers.csv',
            'energy_profiles': self.data_path / 'energy_profiles.parquet',
            'cypher_commands': self.data_path / 'kg_cypher_commands.txt'
        }
        
        availability = {}
        for name, path in required_files.items():
            availability[name] = path.exists()
            if path.exists():
                logger.info(f"✓ Found {name}: {path}")
            else:
                logger.warning(f"✗ Missing {name}: {path}")
                
        return availability
    
    def build_initial_kg(self, clear_existing: bool = True) -> bool:
        """Build initial KG using kg_builder_1"""
        logger.info("="*50)
        logger.info("Building Initial Knowledge Graph")
        logger.info("="*50)
        
        # Check data availability
        data_status = self.check_data_availability()
        
        if not data_status['buildings']:
            logger.error("Cannot build KG: buildings.csv not found")
            logger.info("Run 'python mimic_data_generator.py' first")
            return False
            
        try:
            # Initialize KG builder
            builder = EnergyKnowledgeGraphBuilder(
                uri=self.neo4j_config['uri'],
                user=self.neo4j_config['user'],
                password=self.neo4j_config['password']
            )
            
            # Clear existing data if requested
            if clear_existing:
                logger.info("Clearing existing graph...")
                builder.clear_database()
            
            # Create schema
            logger.info("Creating schema...")
            builder.create_schema()
            
            # Load buildings
            logger.info("Loading buildings...")
            buildings_df = pd.read_csv(self.data_path / 'buildings.csv')
            builder.create_building_nodes(buildings_df)
            
            # Load LV networks if available
            if data_status['lv_networks']:
                logger.info("Loading LV networks...")
                lv_df = pd.read_csv(self.data_path / 'lv_networks.csv')
                builder.create_lv_network_nodes(lv_df)
            
            # Load MV transformers if available
            if data_status['mv_transformers']:
                logger.info("Loading MV transformers...")
                mv_df = pd.read_csv(self.data_path / 'mv_transformers.csv')
                builder.create_mv_transformer_nodes(mv_df)
            
            # Create existing assets
            logger.info("Creating asset nodes...")
            builder.create_existing_assets()
            
            # Create deployment opportunities
            logger.info("Creating deployment opportunities...")
            builder.create_deployment_opportunities()
            
            # Load energy profiles if available
            if data_status['energy_profiles']:
                logger.info("Loading energy profiles...")
                energy_df = pd.read_parquet(self.data_path / 'energy_profiles.parquet')
                builder.create_temporal_nodes(energy_df)
            
            # Get statistics
            stats = builder.get_statistics()
            logger.info(f"KG Built: {stats['total_nodes']} nodes created")
            
            builder.close()
            return True
            
        except Exception as e:
            logger.error(f"Error building KG: {e}")
            return False
    
    def add_adjacency_relationships(self) -> bool:
        """Add adjacency relationships using kg_builder_2"""
        logger.info("="*50)
        logger.info("Adding Adjacency Relationships")
        logger.info("="*50)
        
        try:
            # Initialize adjacency updater
            updater = AdjacencyUpdater(
                uri=self.neo4j_config['uri'],
                user=self.neo4j_config['user'],
                password=self.neo4j_config['password']
            )
            
            # Check KG status
            status = updater.check_kg_status()
            logger.info(f"Found {status['building_count']} buildings in KG")
            
            if status['has_adjacencies']:
                logger.info("Adjacency relationships already exist")
                response = input("Update existing adjacencies? (y/n): ")
                if response.lower() != 'y':
                    return True
            
            # Load buildings data for shared wall info
            buildings_df = pd.read_csv(self.data_path / 'buildings.csv')
            
            # Update shared wall data
            logger.info("Updating shared wall data...")
            updater.update_shared_wall_data(buildings_df)
            
            # Create adjacency relationships
            logger.info("Creating adjacency relationships...")
            stats = updater.create_adjacency_relationships()
            logger.info(f"Created {stats['relationships_created']} adjacency relationships")
            
            # Create thermal clusters
            logger.info("Creating thermal clusters...")
            clusters = updater.create_thermal_clusters()
            logger.info(f"Created {len(clusters)} thermal clusters")
            
            # Run validation
            logger.info("Validating adjacencies...")
            validation = updater.validate_adjacencies()
            logger.info(f"Validation complete: {validation['valid_count']} valid adjacencies")
            
            updater.close()
            return True
            
        except Exception as e:
            logger.error(f"Error adding adjacencies: {e}")
            return False
    
    def rebuild_kg_complete(self) -> bool:
        """Complete rebuild: initial KG + adjacencies"""
        logger.info("="*50)
        logger.info("COMPLETE KG REBUILD")
        logger.info("="*50)
        
        # Build initial KG
        if not self.build_initial_kg(clear_existing=True):
            return False
        
        # Add adjacencies
        if not self.add_adjacency_relationships():
            return False
        
        logger.info("="*50)
        logger.info("✅ Complete KG rebuild successful!")
        logger.info("="*50)
        return True


def integrate_kg_builders_in_main(main_instance):
    """
    Function to integrate KG builders into existing main.py
    Add this to the EnergyGNNSystem class in main.py
    """
    
    def rebuild_knowledge_graph(self, use_adjacency: bool = True):
        """Rebuild the entire knowledge graph from CSV files"""
        
        kg_manager = KGManager(self.config)
        
        # Check if we have the required data
        data_status = kg_manager.check_data_availability()
        
        if not any(data_status.values()):
            self.logger.error("No data files found. Running data generator...")
            os.system("python mimic_data_generator.py")
            
        # Build initial KG
        self.logger.info("Building initial Knowledge Graph...")
        success = kg_manager.build_initial_kg(clear_existing=True)
        
        if not success:
            self.logger.error("Failed to build initial KG")
            return False
        
        # Add adjacencies if requested
        if use_adjacency:
            self.logger.info("Adding adjacency relationships...")
            success = kg_manager.add_adjacency_relationships()
            
        # Reconnect main system to new KG
        self.connect_kg()
        
        return success
    
    # Add the method to main_instance
    main_instance.rebuild_knowledge_graph = rebuild_knowledge_graph
    
    return main_instance


# Add command line interface
def add_kg_commands_to_argparse(parser):
    """Add KG-specific commands to main.py argument parser"""
    
    # Add subcommand for KG operations
    kg_parser = parser.add_parser('kg', help='Knowledge Graph operations')
    kg_subparsers = kg_parser.add_subparsers(dest='kg_command')
    
    # Build command
    build_parser = kg_subparsers.add_parser('build', help='Build KG from CSV files')
    build_parser.add_argument('--clear', action='store_true', 
                            help='Clear existing graph first')
    build_parser.add_argument('--no-adjacency', action='store_true',
                            help='Skip adjacency relationships')
    
    # Update command
    update_parser = kg_subparsers.add_parser('update', help='Update existing KG')
    update_parser.add_argument('--adjacency', action='store_true',
                             help='Update adjacency relationships')
    
    # Status command
    status_parser = kg_subparsers.add_parser('status', help='Check KG status')
    
    return parser


# Usage in main.py
if __name__ == "__main__":
    # Example integration
    
    # 1. Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 2. Create KG manager
    kg_manager = KGManager(config)
    
    # 3. Check what data we have
    print("\nChecking data availability...")
    data_status = kg_manager.check_data_availability()
    
    # 4. Build KG if needed
    if not all(data_status.values()):
        print("\nSome data files are missing.")
        response = input("Generate missing data? (y/n): ")
        if response.lower() == 'y':
            os.system("python mimic_data_generator.py")
    
    # 5. Build or rebuild KG
    print("\nKnowledge Graph Options:")
    print("1. Build initial KG (kg_builder_1)")
    print("2. Add adjacency relationships (kg_builder_2)")
    print("3. Complete rebuild (both)")
    print("4. Skip")
    
    choice = input("\nSelect option (1-4): ")
    
    if choice == '1':
        kg_manager.build_initial_kg()
    elif choice == '2':
        kg_manager.add_adjacency_relationships()
    elif choice == '3':
        kg_manager.rebuild_kg_complete()
    else:
        print("Skipping KG operations")
    
    print("\n✅ KG operations complete!")