"""
Main entry point for Grid Analysis Workflow
"""
import argparse
import sys
import logging
from datetime import datetime
from grid_analysis import GridAnalysisWorkflow
from db_config import DatabaseConfig

def setup_logging(log_file: str = None):
    """Configure logging"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        # Add encoding='utf-8' for the file handler
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=handlers,
        encoding='utf-8'  # Add this line
    )

def get_user_input():
    """Interactive mode to get user input"""
    print("=" * 60)
    print("GRID ANALYSIS WORKFLOW - INTERACTIVE MODE")
    print("=" * 60)
    
    # Get building IDs
    while True:
        try:
            building1 = input("Enter first building OGC FID: ").strip()
            building1_id = int(building1)
            break
        except ValueError:
            print("Invalid input. Please enter a valid integer.")
    
    while True:
        try:
            building2 = input("Enter second building OGC FID: ").strip()
            building2_id = int(building2)
            break
        except ValueError:
            print("Invalid input. Please enter a valid integer.")
    
    # Get prefix
    while True:
        prefix = input("Enter table prefix (e.g., 'run01', 'test_area1'): ").strip()
        if prefix and prefix.replace('_', '').replace('-', '').isalnum():
            break
        print("Invalid prefix. Use only letters, numbers, underscores, and hyphens.")
    
    # Ask about clean start
    clean_response = input("Clean existing tables with this prefix? (y/n) [y]: ").strip().lower()
    clean_start = clean_response != 'n'
    
    return building1_id, building2_id, prefix, clean_start

def run_batch_mode(config_file: str):
    """Run multiple analyses from a configuration file"""
    import json
    
    print(f"Running batch mode with config: {config_file}")
    
    try:
        with open(config_file, 'r') as f:
            configs = json.load(f)
        
        results = []
        for i, config in enumerate(configs, 1):
            print(f"\n{'=' * 60}")
            print(f"Running analysis {i}/{len(configs)}")
            print(f"{'=' * 60}")
            
            workflow = GridAnalysisWorkflow(
                prefix=config['prefix'],
                building1_id=config['building1_id'],
                building2_id=config['building2_id'],
                clean_start=config.get('clean_start', True)
            )
            
            success = workflow.run_complete_workflow()
            results.append({
                'prefix': config['prefix'],
                'success': success
            })
        
        # Print summary
        print(f"\n{'=' * 60}")
        print("BATCH PROCESSING SUMMARY")
        print(f"{'=' * 60}")
        for result in results:
            status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
            print(f"{result['prefix']}: {status}")
        
    except Exception as e:
        print(f"Error in batch mode: {e}")
        sys.exit(1)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Grid Analysis Workflow')
    
    parser.add_argument('--building1', type=int, help='First building OGC FID')
    parser.add_argument('--building2', type=int, help='Second building OGC FID')
    parser.add_argument('--prefix', type=str, help='Table prefix for this run')
    parser.add_argument('--clean', action='store_true', default=True,
                       help='Clean existing tables with this prefix')
    parser.add_argument('--no-clean', dest='clean', action='store_false',
                       help='Keep existing tables')
    parser.add_argument('--batch', type=str, help='Run batch mode with config file')
    parser.add_argument('--log-file', type=str, help='Log file path')
    parser.add_argument('--list-buildings', action='store_true',
                       help='List sample building IDs from database')
    parser.add_argument('--test', action='store_true',
                       help='Run with test building IDs')
    
    args = parser.parse_args()
    
    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = args.log_file or f"grid_analysis_{timestamp}.log"
    setup_logging(log_file)
    
    logger = logging.getLogger(__name__)
    
    # Handle special modes
    if args.list_buildings:
        print("Fetching sample building IDs...")
        db = DatabaseConfig()
        query = """
            SELECT ogc_fid, wijknaam, buurtnaam, woningtype
            FROM amin.buildings_1_deducted
            WHERE pand_geom IS NOT NULL
            ORDER BY RANDOM()
            LIMIT 20
        """
        results = db.execute_query(query, fetch=True)
        print(f"\n{'OGC FID':<10} {'District':<20} {'Neighborhood':<20} {'Type':<20}")
        print("-" * 70)
        for row in results:
            print(f"{row[0]:<10} {(row[1] or 'N/A'):<20} {(row[2] or 'N/A'):<20} {(row[3] or 'N/A'):<20}")
        db.disconnect()
        return
    
    if args.test:
        # Use test building IDs
        building1_id = 4804870
        building2_id = 4794514
        prefix = f"test_{timestamp[:8]}"
        clean_start = True
        print(f"Running test with buildings {building1_id} and {building2_id}")
    
    elif args.batch:
        # Run batch mode
        run_batch_mode(args.batch)
        return
    
    elif args.building1 and args.building2 and args.prefix:
        # Use command line arguments
        building1_id = args.building1
        building2_id = args.building2
        prefix = args.prefix
        clean_start = args.clean
    
    else:
        # Interactive mode
        building1_id, building2_id, prefix, clean_start = get_user_input()
    
    # Run the workflow
    print(f"\nLog file: {log_file}")
    print("Starting workflow...\n")
    
    workflow = GridAnalysisWorkflow(
        prefix=prefix,
        building1_id=building1_id,
        building2_id=building2_id,
        clean_start=clean_start
    )
    
    success = workflow.run_complete_workflow()
    
    if success:
        print(f"\n✓ Workflow completed successfully!")
        print(f"Tables created with prefix: {prefix}")
        print(f"Check log file for details: {log_file}")
    else:
        print(f"\n✗ Workflow failed. Check log file: {log_file}")
        sys.exit(1)

if __name__ == "__main__":
    main()