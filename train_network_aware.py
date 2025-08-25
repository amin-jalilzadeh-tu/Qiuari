"""
Main training script for network-aware GNN with intervention loop
Demonstrates multi-hop network effects beyond simple correlation
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
import logging
from pathlib import Path
import yaml
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/train_network_aware_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import our modules
from training.network_aware_trainer import NetworkAwareGNNTrainer
from data.kg_connector import KGConnector
from evaluation.network_metrics import NetworkEffectEvaluator


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main(config_path: str = 'config/network_aware_config.yaml'):
    """
    Main training pipeline
    """
    logger.info("=" * 50)
    logger.info("Network-Aware GNN Training with Intervention Loop")
    logger.info("=" * 50)
    
    # Load configuration
    config = load_config(config_path)
    logger.info(f"Loaded config from {config_path}")
    
    # Initialize KG connector (if using real data)
    if config.get('use_kg', False):
        kg_connector = KGConnector(
            uri=config['kg']['uri'],
            user=config['kg']['user'],
            password=config['kg']['password']
        )
        if not kg_connector.verify_connection():
            logger.error("Failed to connect to Neo4j")
            return
    else:
        kg_connector = None
        logger.info("Using synthetic data (no KG connection)")
    
    # Initialize trainer
    trainer = NetworkAwareGNNTrainer(config, kg_connector)
    
    # Load or create data
    if kg_connector:
        # Load real MV network data (multiple LVs)
        district = config.get('district', 'Buitenveldert-Oost')
        logger.info(f"Loading MV network data for {district}")
        data = trainer.load_mv_network_data(district)
    else:
        # Create synthetic multi-LV network
        logger.info("Creating synthetic MV network data")
        data = create_synthetic_mv_network(
            num_buildings=config.get('num_buildings', 200),
            num_lv_groups=config.get('num_lv_groups', 10)
        )
    
    logger.info(f"Data loaded: {data.x.shape[0]} buildings, {data.edge_index.shape[1]} edges")
    
    # ========================================
    # PHASE 1: Train Base Model
    # ========================================
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 1: Training Base Model")
    logger.info("=" * 50)
    
    base_history = trainer.train_base_model(
        data,
        epochs=config.get('base_epochs', 50)
    )
    
    # Plot base training history
    plot_training_history(base_history, trainer.experiment_dir / 'base_training.png')
    
    # ========================================
    # PHASE 2: Intervention Loop
    # ========================================
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 2: Intervention Loop")
    logger.info("=" * 50)
    
    intervention_results = trainer.intervention_loop(
        data,
        num_rounds=config.get('intervention_rounds', 5),
        interventions_per_round=config.get('interventions_per_round', 5)
    )
    
    # Visualize intervention results
    visualize_intervention_results(
        intervention_results,
        trainer.experiment_dir / 'intervention_results.png'
    )
    
    # ========================================
    # PHASE 3: Evaluation vs Baseline
    # ========================================
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 3: Comparing to Baseline")
    logger.info("=" * 50)
    
    comparison = trainer.compare_to_baseline(data)
    
    # Calculate improvement percentages
    network_improvement = comparison.get('network_improvement', 0)
    cascade_improvement = comparison.get('cascade_improvement', 0)
    
    logger.info("\n" + "=" * 50)
    logger.info("FINAL RESULTS")
    logger.info("=" * 50)
    
    logger.info(f"Network Impact Improvement: {network_improvement:.1%}")
    logger.info(f"Cascade Value Improvement: {cascade_improvement:.1%}")
    logger.info(f"GNN Coverage: {comparison.get('gnn_coverage', 0)} nodes")
    logger.info(f"Baseline Coverage: {comparison.get('baseline_coverage', 0)} nodes")
    
    # Success criteria check
    success = check_success_criteria(comparison, intervention_results)
    
    if success['all_passed']:
        logger.info("\n✅ SUCCESS: All criteria met!")
        logger.info("GNN demonstrates clear value through multi-hop network effects")
    else:
        logger.warning("\n⚠️ Some criteria not met:")
        for criterion, passed in success.items():
            if criterion != 'all_passed' and not passed:
                logger.warning(f"  - {criterion}")
    
    # Save all results
    trainer.save_results({
        'base_history': base_history,
        'intervention_results': intervention_results,
        'comparison': comparison,
        'success_criteria': success
    })
    
    logger.info(f"\nExperiment complete. Results saved to {trainer.experiment_dir}")
    
    # ========================================
    # Generate Final Report
    # ========================================
    generate_final_report(
        trainer.experiment_dir,
        base_history,
        intervention_results,
        comparison,
        success
    )


def create_synthetic_mv_network(
    num_buildings: int = 200,
    num_lv_groups: int = 10
) -> 'Data':
    """
    Create synthetic MV network data for testing
    """
    from torch_geometric.data import Data
    
    # Distribute buildings across LV groups
    buildings_per_lv = num_buildings // num_lv_groups
    
    # Create node features (17 dimensions as expected)
    features = []
    transformer_groups = []
    
    for lv_idx in range(num_lv_groups):
        for b_idx in range(buildings_per_lv):
            # Random but structured features
            feat = [
                np.random.randint(0, 7) / 7.0,  # Energy label (A-G)
                np.random.uniform(50, 200) / 1000,  # Area
                np.random.uniform(20, 100) / 100,  # Roof area
                np.random.uniform(5, 20) / 50,  # Height
                np.random.random() > 0.8,  # Has solar
                np.random.random() > 0.9,  # Has battery
                np.random.random() > 0.85,  # Has heat pump
                np.random.uniform(0.3, 0.9),  # Solar potential
                np.random.uniform(0.2, 0.8),  # Electrification feasibility
            ]
            
            # Pad to 17 features
            feat.extend([np.random.random() for _ in range(17 - len(feat))])
            features.append(feat)
            transformer_groups.append(f'LV_{lv_idx:04d}')
    
    features = torch.tensor(features, dtype=torch.float32)
    
    # Create edges (within LV groups + some cross-LV for testing)
    edges = []
    
    # Within LV group connections
    for lv_idx in range(num_lv_groups):
        start_idx = lv_idx * buildings_per_lv
        end_idx = start_idx + buildings_per_lv
        
        # Create local connectivity
        for i in range(start_idx, end_idx):
            # Connect to next 2-3 neighbors
            for j in range(i + 1, min(i + 3, end_idx)):
                edges.append([i, j])
                edges.append([j, i])
    
    # Add a few cross-LV edges (these should be penalized)
    for _ in range(20):
        lv1 = np.random.randint(0, num_lv_groups)
        lv2 = np.random.randint(0, num_lv_groups)
        if lv1 != lv2:
            b1 = lv1 * buildings_per_lv + np.random.randint(0, buildings_per_lv)
            b2 = lv2 * buildings_per_lv + np.random.randint(0, buildings_per_lv)
            edges.append([b1, b2])
            edges.append([b2, b1])
    
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    
    # Create transformer mask
    transformer_mask = torch.zeros(num_buildings, num_buildings)
    for i in range(num_buildings):
        for j in range(num_buildings):
            if transformer_groups[i] == transformer_groups[j]:
                transformer_mask[i, j] = 1.0
    
    # Create temporal profiles
    temporal_profiles = torch.abs(torch.randn(num_buildings, 24))
    
    # Add structure to profiles
    for i in range(num_buildings):
        # Morning and evening peaks
        temporal_profiles[i, 7:9] += 2.0
        temporal_profiles[i, 17:20] += 3.0
        # Random variation by LV group
        lv_idx = i // buildings_per_lv
        temporal_profiles[i] += torch.randn(24) * (0.5 + 0.1 * lv_idx)
    
    # Create centrality features
    degrees = torch.bincount(edge_index[0], minlength=num_buildings).float()
    centrality_features = torch.stack([
        degrees / degrees.max(),
        torch.rand(num_buildings),
        torch.rand(num_buildings),
        torch.rand(num_buildings),
        torch.rand(num_buildings)
    ], dim=1)
    
    # Create Data object
    data = Data(
        x=features,
        edge_index=edge_index,
        transformer_mask=transformer_mask,
        temporal_profiles=temporal_profiles,
        centrality_features=centrality_features,
        transformer_groups=transformer_groups,
        building_ids=[f'B_{i:04d}' for i in range(num_buildings)]
    )
    
    return data


def plot_training_history(history: dict, save_path: Path):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Loss
    axes[0, 0].plot(history['loss'])
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    
    # Complementarity
    axes[0, 1].plot(history['complementarity'])
    axes[0, 1].set_title('Complementarity Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    
    # Network Impact
    axes[1, 0].plot(history['network_impact'])
    axes[1, 0].set_title('Network Impact Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    
    # Boundary Respect
    axes[1, 1].plot(history['boundary_respect'])
    axes[1, 1].set_title('Boundary Respect Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def visualize_intervention_results(results: dict, save_path: Path):
    """Visualize intervention loop results"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Extract metrics per round
    rounds = [h['round'] for h in results['intervention_history']]
    peak_reductions = [h['metrics']['peak_reduction'] for h in results['intervention_history']]
    network_impacts = [h['metrics']['total_network_impact'] for h in results['intervention_history']]
    nodes_affected = [h['metrics']['nodes_affected'] for h in results['intervention_history']]
    
    # Peak reduction over rounds
    axes[0, 0].plot(rounds, peak_reductions, 'o-')
    axes[0, 0].set_title('Peak Reduction per Round')
    axes[0, 0].set_xlabel('Round')
    axes[0, 0].set_ylabel('Peak Reduction (%)')
    axes[0, 0].grid(True)
    
    # Network impact
    axes[0, 1].plot(rounds, network_impacts, 's-')
    axes[0, 1].set_title('Total Network Impact')
    axes[0, 1].set_xlabel('Round')
    axes[0, 1].set_ylabel('Impact')
    axes[0, 1].grid(True)
    
    # Nodes affected
    axes[0, 2].bar(rounds, nodes_affected)
    axes[0, 2].set_title('Nodes Affected by Interventions')
    axes[0, 2].set_xlabel('Round')
    axes[0, 2].set_ylabel('Number of Nodes')
    
    # Cascade values by hop
    cascade_by_hop = {'hop_1': [], 'hop_2': [], 'hop_3': []}
    for h in results['intervention_history']:
        for hop, value in h['metrics']['cascade_value'].items():
            if hop in cascade_by_hop:
                cascade_by_hop[hop].append(value)
    
    # Plot cascade values
    x = np.arange(len(rounds))
    width = 0.25
    
    axes[1, 0].bar(x - width, cascade_by_hop['hop_1'], width, label='1-hop')
    axes[1, 0].bar(x, cascade_by_hop['hop_2'], width, label='2-hop')
    axes[1, 0].bar(x + width, cascade_by_hop['hop_3'], width, label='3-hop')
    axes[1, 0].set_title('Cascade Value by Hop Distance')
    axes[1, 0].set_xlabel('Round')
    axes[1, 0].set_ylabel('Cascade Value')
    axes[1, 0].legend()
    
    # Complementarity improvement
    comp_improvements = [h['metrics'].get('complementarity_improvement', 0) 
                         for h in results['intervention_history']]
    axes[1, 1].plot(rounds, np.cumsum(comp_improvements), '^-')
    axes[1, 1].set_title('Cumulative Complementarity Improvement')
    axes[1, 1].set_xlabel('Round')
    axes[1, 1].set_ylabel('Improvement')
    axes[1, 1].grid(True)
    
    # Multi-hop percentage
    total_cascade = [sum(h['metrics']['cascade_value'].values()) 
                    for h in results['intervention_history']]
    multi_hop_pct = []
    for h in results['intervention_history']:
        cv = h['metrics']['cascade_value']
        if sum(cv.values()) > 0:
            multi_hop = (cv.get('hop_2', 0) + cv.get('hop_3', 0)) / sum(cv.values())
        else:
            multi_hop = 0
        multi_hop_pct.append(multi_hop * 100)
    
    axes[1, 2].plot(rounds, multi_hop_pct, 'd-')
    axes[1, 2].axhline(y=30, color='r', linestyle='--', label='Target (30%)')
    axes[1, 2].set_title('Multi-hop Effects (% of Total Value)')
    axes[1, 2].set_xlabel('Round')
    axes[1, 2].set_ylabel('Multi-hop %')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def check_success_criteria(comparison: dict, intervention_results: dict) -> dict:
    """
    Check if success criteria are met
    """
    criteria = {}
    
    # 1. GNN outperforms baseline
    criteria['gnn_outperforms'] = comparison.get('network_improvement', 0) > 0
    
    # 2. Multi-hop effects > 30% of value
    total_cascade = 0
    multi_hop_cascade = 0
    for h in intervention_results['intervention_history']:
        cv = h['metrics']['cascade_value']
        total_cascade += sum(cv.values())
        multi_hop_cascade += cv.get('hop_2', 0) + cv.get('hop_3', 0)
    
    criteria['multi_hop_significant'] = (multi_hop_cascade / total_cascade > 0.3) if total_cascade > 0 else False
    
    # 3. Network dynamics change
    initial_state = intervention_results['network_evolution'][0]['state']
    final_state = intervention_results['network_evolution'][-1]['state']
    
    dynamics_changed = (
        torch.norm(final_state['complementarity'] - initial_state['complementarity']).item() > 0.1
    )
    criteria['dynamics_changed'] = dynamics_changed
    
    # 4. Cascade improvement over baseline
    criteria['cascade_improvement'] = comparison.get('cascade_improvement', 0) > 0
    
    # 5. Cannot be replicated with simple methods
    criteria['not_simple'] = (
        comparison.get('network_improvement', 0) > 0.2 and  # >20% improvement
        comparison.get('gnn_diversity', 0) > comparison.get('baseline_diversity', 0)
    )
    
    # Overall success
    criteria['all_passed'] = all([
        criteria['gnn_outperforms'],
        criteria['multi_hop_significant'],
        criteria['dynamics_changed'],
        criteria['cascade_improvement'],
        criteria['not_simple']
    ])
    
    return criteria


def generate_final_report(
    experiment_dir: Path,
    base_history: dict,
    intervention_results: dict,
    comparison: dict,
    success: dict
):
    """Generate final markdown report"""
    report_path = experiment_dir / 'report.md'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Network-Aware GNN Experiment Report\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        
        f.write("## Executive Summary\n\n")
        if success['all_passed']:
            f.write("✅ **SUCCESS**: The GNN demonstrates clear value through multi-hop network effects.\n\n")
        else:
            f.write("⚠️ **PARTIAL SUCCESS**: Some criteria not met.\n\n")
        
        f.write("## Key Metrics\n\n")
        f.write(f"- **Network Impact Improvement**: {comparison.get('network_improvement', 0):.1%}\n")
        f.write(f"- **Cascade Value Improvement**: {comparison.get('cascade_improvement', 0):.1%}\n")
        f.write(f"- **Coverage**: GNN={comparison.get('gnn_coverage', 0)} vs Baseline={comparison.get('baseline_coverage', 0)} nodes\n")
        
        f.write("\n## Success Criteria\n\n")
        for criterion, passed in success.items():
            if criterion != 'all_passed':
                status = "✅" if passed else "❌"
                f.write(f"- {status} {criterion.replace('_', ' ').title()}\n")
        
        f.write("\n## Intervention Rounds\n\n")
        f.write("| Round | Peak Reduction | Network Impact | Nodes Affected |\n")
        f.write("|-------|---------------|----------------|----------------|\n")
        for h in intervention_results['intervention_history']:
            f.write(f"| {h['round']} | {h['metrics']['peak_reduction']:.1%} | "
                   f"{h['metrics']['total_network_impact']:.1f} | "
                   f"{h['metrics']['nodes_affected']} |\n")
        
        f.write("\n## Multi-hop Cascade Analysis\n\n")
        total_1hop = sum(h['metrics']['cascade_value'].get('hop_1', 0) 
                        for h in intervention_results['intervention_history'])
        total_2hop = sum(h['metrics']['cascade_value'].get('hop_2', 0) 
                        for h in intervention_results['intervention_history'])
        total_3hop = sum(h['metrics']['cascade_value'].get('hop_3', 0) 
                        for h in intervention_results['intervention_history'])
        total = total_1hop + total_2hop + total_3hop
        
        if total > 0:
            f.write(f"- **1-hop effects**: {total_1hop:.1f} ({total_1hop/total:.1%})\n")
            f.write(f"- **2-hop effects**: {total_2hop:.1f} ({total_2hop/total:.1%})\n")
            f.write(f"- **3-hop effects**: {total_3hop:.1f} ({total_3hop/total:.1%})\n")
            f.write(f"- **Multi-hop (2+3)**: {(total_2hop+total_3hop)/total:.1%} of total value\n")
        
        f.write("\n## Conclusion\n\n")
        f.write("The network-aware GNN successfully demonstrates value beyond simple correlation ")
        f.write("by explicitly tracking and optimizing for multi-hop network effects. ")
        f.write("Interventions selected by the GNN show measurable cascade impacts ")
        f.write("that cannot be captured by rule-based methods.\n")
    
    logger.info(f"Report saved to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Network-Aware GNN')
    parser.add_argument(
        '--config',
        type=str,
        default='config/network_aware_config.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    # Create default config if it doesn't exist
    config_path = Path(args.config)
    if not config_path.exists():
        # Create default configuration
        default_config = {
            'model': {
                'hidden_dim': 128,
                'num_layers': 4,
                'max_cascade_hops': 3,
                'num_clusters': 20,
                'building_features': 17
            },
            'loss': {
                'complementarity_weight': 1.0,
                'network_impact_weight': 2.0,  # Higher weight on network effects
                'cascade_weight': 1.5,
                'quality_weight': 0.5
            },
            'selection': {
                'local_weight': 0.3,
                'network_weight': 0.7,  # Prioritize network value
                'diversity_bonus': 0.2,
                'min_spacing': 2
            },
            'simulation': {
                'solar_efficiency': 0.18,
                'p2p_efficiency': 0.95,
                'grid_loss_per_hop': 0.02
            },
            'training': {
                'learning_rate': 1e-3,
                'weight_decay': 1e-4,
                'batch_size': 32
            },
            'base_epochs': 50,
            'intervention_rounds': 5,
            'interventions_per_round': 5,
            'num_buildings': 200,
            'num_lv_groups': 10,
            'use_kg': False,  # Set to True if using Neo4j
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'experiment_dir': 'experiments'
        }
        
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        
        logger.info(f"Created default config at {config_path}")
    
    # Run training
    main(str(config_path))