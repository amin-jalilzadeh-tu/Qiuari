"""
Network-aware GNN trainer with intervention loop
Demonstrates multi-hop value beyond simple correlation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm

# Import our custom modules
from models.network_aware_layers import NetworkAwareGNN
from training.network_aware_loss import NetworkAwareDiscoveryLoss, CascadePredictionLoss
from tasks.intervention_selection import NetworkAwareInterventionSelector
from simulation.simple_intervention import SimpleInterventionSimulator
from data.kg_connector import KGConnector

logger = logging.getLogger(__name__)


class NetworkAwareGNNTrainer:
    """
    Trains GNN to understand multi-hop network effects
    Includes intervention loop with cascade tracking
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        kg_connector: Optional[KGConnector] = None
    ):
        """
        Initialize trainer
        
        Args:
            config: Training configuration
            kg_connector: Connection to knowledge graph
        """
        self.config = config
        self.kg_connector = kg_connector
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Initialize model
        self.model = NetworkAwareGNN(config['model']).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-3),
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
        
        # Initialize losses
        self.discovery_loss = NetworkAwareDiscoveryLoss(config.get('loss', {}))
        self.cascade_loss = CascadePredictionLoss()
        
        # Initialize intervention components
        self.selector = NetworkAwareInterventionSelector(config.get('selection', {}))
        self.simulator = SimpleInterventionSimulator(config.get('simulation', {}))
        
        # Training state
        self.current_epoch = 0
        self.intervention_round = 0
        self.best_loss = float('inf')
        self.training_history = []
        
        # Experiment tracking
        self.experiment_dir = Path(config.get('experiment_dir', 'experiments')) / f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized NetworkAwareGNNTrainer on {self.device}")
        logger.info(f"Experiment directory: {self.experiment_dir}")
    
    def load_mv_network_data(self, district_name: str) -> Data:
        """
        Load multi-LV network data (MV level)
        
        Args:
            district_name: District to load
            
        Returns:
            PyG Data object with ~200 buildings across multiple LVs
        """
        if self.kg_connector is None:
            raise ValueError("KG connector required for loading network data")
        
        # Get all LV groups in district
        lv_groups = self.kg_connector.get_lv_groups_in_district(district_name)
        logger.info(f"Found {len(lv_groups)} LV groups in {district_name}")
        
        all_buildings = []
        all_edges = []
        node_offset = 0
        transformer_boundaries = []
        
        for lv_group in lv_groups[:10]:  # Limit to 10 LV groups for ~200 buildings
            # Get buildings in this LV group
            buildings = self.kg_connector.get_buildings_by_cable_group(lv_group)
            
            if len(buildings) < 3:
                continue
            
            # Track transformer boundary
            transformer_boundaries.extend([lv_group] * len(buildings))
            
            # Create edges within LV group (full connectivity for now)
            n_buildings = len(buildings)
            for i in range(n_buildings):
                for j in range(i + 1, min(i + 3, n_buildings)):  # Connect to next 2 buildings
                    all_edges.append([node_offset + i, node_offset + j])
                    all_edges.append([node_offset + j, node_offset + i])  # Bidirectional
            
            all_buildings.extend(buildings)
            node_offset += n_buildings
        
        logger.info(f"Loaded {len(all_buildings)} buildings total")
        
        # Create node features
        node_features = self._create_node_features(all_buildings)
        
        # Create edge index
        edge_index = torch.tensor(all_edges, dtype=torch.long).t()
        
        # Create transformer mask (1 if same transformer, 0 otherwise)
        n_nodes = len(all_buildings)
        transformer_mask = torch.zeros(n_nodes, n_nodes)
        for i in range(n_nodes):
            for j in range(n_nodes):
                if transformer_boundaries[i] == transformer_boundaries[j]:
                    transformer_mask[i, j] = 1.0
        
        # Get temporal profiles if available
        building_ids = [b['id'] for b in all_buildings]
        temporal_profiles = self._load_temporal_profiles(building_ids)
        
        # Create centrality features
        centrality_features = self._calculate_centrality_features(edge_index, n_nodes)
        
        # Create Data object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            transformer_mask=transformer_mask,
            temporal_profiles=temporal_profiles,
            centrality_features=centrality_features,
            building_ids=building_ids,
            transformer_groups=transformer_boundaries
        )
        
        return data
    
    def create_network_aware_labels(self, data: Data) -> Dict[str, torch.Tensor]:
        """
        Create network-aware labels for semi-supervised learning
        Labels based on network impact, not just local quality
        
        Args:
            data: Network data
            
        Returns:
            Dictionary of network-aware labels
        """
        labels = {}
        
        # 1. Network centrality labels (buildings with high betweenness)
        centrality = data.centrality_features[:, 0]  # Degree centrality
        labels['high_centrality'] = centrality > centrality.quantile(0.8)
        
        # 2. Boundary nodes (connect different transformers)
        boundary_nodes = self._identify_boundary_nodes(
            data.edge_index, 
            data.transformer_groups
        )
        labels['boundary_nodes'] = boundary_nodes
        
        # 3. High cascade potential (based on neighborhood diversity)
        cascade_potential = self._calculate_cascade_potential_labels(
            data.x, data.edge_index
        )
        labels['high_cascade'] = cascade_potential > cascade_potential.quantile(0.7)
        
        # 4. Complementarity hubs (nodes with diverse neighbors)
        comp_hubs = self._identify_complementarity_hubs(
            data.temporal_profiles, data.edge_index
        )
        labels['comp_hubs'] = comp_hubs
        
        return labels
    
    def train_base_model(self, data: Data, epochs: int = 50) -> Dict[str, List[float]]:
        """
        Phase 1: Train base model to understand network patterns
        
        Args:
            data: Network data
            epochs: Number of training epochs
            
        Returns:
            Training history
        """
        logger.info("Starting Phase 1: Base model training")
        
        # Create network-aware labels
        labels = self.create_network_aware_labels(data)
        
        # Move data to device
        data = data.to(self.device)
        for key in labels:
            labels[key] = labels[key].to(self.device)
        
        history = {
            'loss': [],
            'complementarity': [],
            'network_impact': [],
            'boundary_respect': []
        }
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Training step
            self.model.train()
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(
                data.x,
                data.edge_index,
                centrality_features=data.centrality_features,
                boundary_mask=labels.get('boundary_nodes'),
                grid_level=torch.zeros(data.x.size(0), device=self.device, dtype=torch.long)  # All building level
            )
            
            # Prepare network data for loss
            network_data = {
                'temporal_profiles': data.temporal_profiles,
                'edge_index': data.edge_index,
                'transformer_mask': data.transformer_mask
            }
            
            # Calculate loss
            loss, loss_components = self.discovery_loss(
                outputs,
                network_data
            )
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Log metrics
            history['loss'].append(loss.item())
            comp_val = loss_components.get('complementarity', torch.tensor(0))
            history['complementarity'].append(comp_val.item() if hasattr(comp_val, 'item') else comp_val)
            net_val = loss_components.get('network_impact', torch.tensor(0))
            history['network_impact'].append(net_val.item() if hasattr(net_val, 'item') else net_val)
            bound_val = loss_components.get('boundary', 0)
            history['boundary_respect'].append(bound_val.item() if hasattr(bound_val, 'item') else bound_val)
            
            if epoch % 10 == 0:
                comp_loss = loss_components.get('complementarity', torch.tensor(0))
                comp_val = comp_loss.item() if hasattr(comp_loss, 'item') else float(comp_loss)
                net_loss = loss_components.get('network_impact', torch.tensor(0))
                net_val = net_loss.item() if hasattr(net_loss, 'item') else float(net_loss)
                logger.info(f"Epoch {epoch}: Loss={loss.item():.4f}, "
                          f"Comp={comp_val:.4f}, "
                          f"Network={net_val:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step(loss)
            
            # Save best model
            if loss < self.best_loss:
                self.best_loss = loss
                self.save_checkpoint('best_base_model.pt')
        
        logger.info("Phase 1 complete")
        return history
    
    def intervention_loop(
        self,
        data: Data,
        num_rounds: int = 5,
        interventions_per_round: int = 5
    ) -> Dict[str, Any]:
        """
        Phase 2: Iterative intervention and learning
        
        Args:
            data: Network data
            num_rounds: Number of intervention rounds
            interventions_per_round: Interventions per round
            
        Returns:
            Intervention results and metrics
        """
        logger.info("Starting Phase 2: Intervention loop")
        
        # Move data to device
        data = data.to(self.device)
        
        # Initialize network state
        network_state = self._initialize_network_state(data)
        
        # Track interventions and results
        intervention_history = []
        cascade_metrics = []
        network_evolution = []
        
        # Store initial state
        network_evolution.append({
            'round': 0,
            'state': {k: v.clone() if isinstance(v, torch.Tensor) else v 
                     for k, v in network_state.items()}
        })
        
        for round_idx in range(num_rounds):
            self.intervention_round = round_idx
            logger.info(f"\n=== Intervention Round {round_idx + 1} ===")
            
            # Step 1: GNN selects interventions
            selected_nodes = self.select_interventions(
                data, 
                network_state,
                k=interventions_per_round,
                existing_interventions=[h['nodes'] for h in intervention_history]
            )
            
            logger.info(f"Selected nodes: {selected_nodes}")
            
            # Step 2: Simulate interventions and track cascades
            cascade_effects = self.simulate_interventions(
                selected_nodes,
                data,
                network_state
            )
            
            # Step 3: Update GNN with cascade knowledge
            self.update_model_with_cascades(
                data,
                selected_nodes,
                cascade_effects
            )
            
            # Step 4: Update network state
            new_network_state = self.simulator.update_network_state(
                network_state,
                cascade_effects
            )
            
            # Step 5: Re-cluster with new network state
            new_clusters = self.cluster_network(data, new_network_state)
            
            # Step 6: Evaluate round
            metrics = self.evaluate_round(
                selected_nodes,
                cascade_effects,
                network_state,
                new_network_state,
                new_clusters
            )
            
            # Record history
            intervention_history.append({
                'round': round_idx + 1,
                'nodes': selected_nodes,
                'metrics': metrics
            })
            
            cascade_metrics.append(cascade_effects)
            
            network_evolution.append({
                'round': round_idx + 1,
                'state': {k: v.clone() if isinstance(v, torch.Tensor) else v 
                         for k, v in new_network_state.items()},
                'clusters': new_clusters
            })
            
            # Update state for next round
            network_state = new_network_state
            
            # Log round results
            logger.info(f"Round {round_idx + 1} metrics:")
            logger.info(f"  - Peak reduction: {metrics['peak_reduction']:.2%}")
            logger.info(f"  - Network impact: {metrics['total_network_impact']:.2f}")
            logger.info(f"  - Cascade value: {sum(metrics['cascade_value'].values()):.2f}")
        
        logger.info("Phase 2 complete")
        
        return {
            'intervention_history': intervention_history,
            'cascade_metrics': cascade_metrics,
            'network_evolution': network_evolution,
            'final_state': network_state
        }
    
    def select_interventions(
        self,
        data: Data,
        network_state: Dict[str, torch.Tensor],
        k: int = 5,
        existing_interventions: Optional[List[List[int]]] = None
    ) -> List[int]:
        """
        GNN-based intervention selection
        """
        self.model.eval()
        
        with torch.no_grad():
            # Prepare additional features (ensure on correct device)
            boundary_mask = torch.zeros(data.x.size(0), device=self.device)
            grid_level = torch.zeros(data.x.size(0), dtype=torch.long, device=self.device)
            
            # Get GNN outputs
            outputs = self.model(
                data.x,
                data.edge_index,
                centrality_features=data.centrality_features,
                boundary_mask=boundary_mask,
                grid_level=grid_level
            )
            
            # Rank interventions
            ranking_scores = self.selector.rank_interventions(
                outputs,
                data.x,
                data.edge_index,
                existing_interventions=sum(existing_interventions, []) if existing_interventions else None
            )
            
            # Select optimal set
            selected = self.selector.select_optimal_set(
                ranking_scores,
                k=k,
                edge_index=data.edge_index
            )
        
        return selected
    
    def simulate_interventions(
        self,
        selected_nodes: List[int],
        data: Data,
        network_state: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Simulate cascade effects of interventions
        """
        # Get building features for selected nodes with more realistic data
        building_features = {}
        for idx in selected_nodes:
            # Extract features from node feature tensor
            # Assuming: [energy_label, area, roof_area, height, has_solar, ...]
            building_features[idx] = {
                'suitable_roof_area': max(20.0, data.x[idx, 2].item() * 100),  # Min 20m²
                'orientation': 'south',  # Assume optimal orientation
                'shading': 0.1,  # 10% shading factor
                'energy_label': chr(65 + min(6, max(0, int(data.x[idx, 0].item())))),  # A-G
                'area': data.x[idx, 1].item() * 1000,  # Building area in m²
                'height': data.x[idx, 3].item() * 50 if data.x.shape[1] > 3 else 10  # Height
            }
        
        # Simulate intervention round with proper building features
        _, metrics = self.simulator.simulate_intervention_round(
            selected_nodes,
            network_state,
            building_features,
            data.edge_index
        )
        
        # Also get detailed cascade effects for each node
        cascade_effects = {}
        for node in selected_nodes:
            # Generate realistic solar profile based on building features
            solar_result = self.simulator.add_solar(
                building_features[node],
                time_series=self._generate_solar_irradiance_profile()
            )
            
            intervention = {
                'building_id': node,
                'type': 'solar',
                'generation_profile': torch.tensor(
                    solar_result['generation_profile'], 
                    device=self.device
                )
            }
            
            cascade = self.simulator.calculate_cascade_effects(
                intervention,
                network_state,
                data.edge_index,
                max_hops=3
            )
            
            # Aggregate cascades
            for hop_key, effects in cascade.items():
                if hop_key not in cascade_effects:
                    cascade_effects[hop_key] = {
                        'energy_impact': torch.zeros_like(network_state['demand']),
                        'congestion_relief': torch.zeros_like(network_state.get('congestion', network_state['demand'])),
                        'economic_value': torch.zeros_like(network_state['demand'])
                    }
                
                for effect_type, values in effects.items():
                    # Ensure values are on same device before accumulating
                    if isinstance(values, torch.Tensor):
                        values = values.to(cascade_effects[hop_key][effect_type].device)
                    cascade_effects[hop_key][effect_type] = cascade_effects[hop_key][effect_type] + values
        
        return cascade_effects
    
    def update_model_with_cascades(
        self,
        data: Data,
        selected_nodes: List[int],
        cascade_effects: Dict[str, torch.Tensor]
    ):
        """
        Fine-tune model with observed cascade effects
        """
        self.model.train()
        
        # Create intervention mask
        intervention_mask = torch.zeros(data.x.size(0), device=self.device)
        intervention_mask[selected_nodes] = 1.0
        
        # Prepare additional features (ensure on correct device)
        boundary_mask = torch.zeros(data.x.size(0), device=self.device)
        grid_level = torch.zeros(data.x.size(0), dtype=torch.long, device=self.device)
        
        # Forward pass with intervention mask
        outputs = self.model(
            data.x,
            data.edge_index,
            intervention_mask=intervention_mask,
            centrality_features=data.centrality_features,
            boundary_mask=boundary_mask,
            grid_level=grid_level
        )
        
        # Calculate cascade prediction loss
        if 'cascade_effects' in outputs:
            loss, _ = self.cascade_loss(
                outputs['cascade_effects'],
                cascade_effects,
                intervention_mask
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            logger.debug(f"Cascade prediction loss: {loss.item():.4f}")
    
    def cluster_network(
        self,
        data: Data,
        network_state: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Re-cluster network with updated state
        """
        self.model.eval()
        
        with torch.no_grad():
            # Update features with new network state
            updated_features = self._update_features_with_state(
                data.x, network_state
            )
            
            # Prepare additional features
            boundary_mask = torch.zeros(updated_features.size(0), device=self.device)
            grid_level = torch.zeros(updated_features.size(0), dtype=torch.long, device=self.device)
            
            # Get new clustering
            outputs = self.model(
                updated_features,
                data.edge_index,
                centrality_features=data.centrality_features,
                boundary_mask=boundary_mask,
                grid_level=grid_level
            )
            
            # Extract clusters
            if 'clusters' in outputs:
                clusters = outputs['clusters']
            else:
                # Fallback: use embeddings for clustering
                from sklearn.cluster import KMeans
                embeddings = outputs['embeddings'].cpu().numpy()
                kmeans = KMeans(n_clusters=min(20, len(embeddings) // 5))
                clusters = torch.tensor(kmeans.fit_predict(embeddings))
        
        return clusters
    
    def evaluate_round(
        self,
        selected_nodes: List[int],
        cascade_effects: Dict[str, Any],
        old_state: Dict[str, torch.Tensor],
        new_state: Dict[str, torch.Tensor],
        new_clusters: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Evaluate intervention round
        """
        metrics = {}
        
        # Peak reduction
        old_peak = old_state['net_demand'].max().item()
        new_peak = new_state['net_demand'].max().item()
        metrics['peak_reduction'] = (old_peak - new_peak) / old_peak if old_peak > 0 else 0
        
        # Total network impact
        total_impact = 0
        cascade_value = {}
        
        for hop_key, effects in cascade_effects.items():
            if 'hop_' in hop_key:
                hop_impact = effects['energy_impact'].sum().item()
                total_impact += hop_impact
                cascade_value[hop_key] = hop_impact
        
        metrics['total_network_impact'] = total_impact
        metrics['cascade_value'] = cascade_value
        
        # Cluster quality improvement
        old_complementarity = old_state.get('complementarity', torch.tensor(0)).mean().item()
        new_complementarity = new_state.get('complementarity', torch.tensor(0)).mean().item()
        metrics['complementarity_improvement'] = new_complementarity - old_complementarity
        
        # Number of affected nodes
        affected_nodes = sum([
            (effects['energy_impact'] > 0).sum().item()
            for effects in cascade_effects.values()
            if isinstance(effects, dict) and 'energy_impact' in effects
        ])
        metrics['nodes_affected'] = affected_nodes
        
        return metrics
    
    def compare_to_baseline(self, data: Data) -> Dict[str, Any]:
        """
        Compare GNN selection to simple baseline
        """
        logger.info("\n=== Comparing to baseline ===")
        
        # Initialize network state
        network_state = self._initialize_network_state(data)
        
        # GNN selection
        gnn_selected = self.select_interventions(
            data, network_state, k=10
        )
        
        # Baseline selection (by energy label and roof area)
        baseline_selected = self._baseline_selection(data, k=10)
        
        # Evaluate both
        comparison = self.selector.evaluate_selection(
            gnn_selected,
            self.model(data.x, data.edge_index),
            baseline_selected,
            data.edge_index
        )
        
        logger.info("Comparison results:")
        for key, value in comparison.items():
            logger.info(f"  {key}: {value:.4f}")
        
        return comparison
    
    def _create_node_features(self, buildings: List[Dict]) -> torch.Tensor:
        """Create node feature tensor from building data"""
        features = []
        
        for b in buildings:
            # Extract and normalize features
            feat = [
                ord(b.get('energy_label', 'D')) - ord('A'),  # Energy label as ordinal
                b.get('area', 100) / 1000,  # Normalize area
                b.get('roof_area', 50) / 100,  # Normalize roof area
                b.get('height', 10) / 50,  # Normalize height
                1.0 if b.get('has_solar') else 0.0,
                1.0 if b.get('has_battery') else 0.0,
                1.0 if b.get('has_heat_pump') else 0.0,
                b.get('solar_potential', 0.5),
                b.get('electrification', 0.5),
                # Add more features as needed
            ]
            
            # Pad to expected dimension
            while len(feat) < 17:
                feat.append(0.0)
            
            features.append(feat[:17])
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _load_temporal_profiles(self, building_ids: List[str]) -> torch.Tensor:
        """Load temporal profiles for buildings"""
        # For now, generate synthetic profiles
        n_buildings = len(building_ids)
        n_timesteps = 24  # Hourly for one day
        
        # Generate diverse profiles
        profiles = torch.randn(n_buildings, n_timesteps)
        
        # Add some structure (peak hours, etc.)
        for i in range(n_buildings):
            # Morning peak
            profiles[i, 7:9] += 1.0
            # Evening peak
            profiles[i, 17:20] += 1.5
            # Night time low
            profiles[i, 0:6] *= 0.3
        
        return torch.abs(profiles)  # Ensure positive
    
    def _calculate_centrality_features(
        self,
        edge_index: torch.Tensor,
        n_nodes: int
    ) -> torch.Tensor:
        """Calculate various centrality measures"""
        features = torch.zeros(n_nodes, 5)
        
        # Degree centrality
        degrees = torch.bincount(edge_index[0], minlength=n_nodes).float()
        features[:, 0] = degrees / degrees.max() if degrees.max() > 0 else degrees
        
        # Add more centrality measures as needed
        # For now, repeat degree with some noise
        for i in range(1, 5):
            features[:, i] = features[:, 0] + torch.randn(n_nodes) * 0.1
        
        return features
    
    def _identify_boundary_nodes(
        self,
        edge_index: torch.Tensor,
        transformer_groups: List[str]
    ) -> torch.Tensor:
        """Identify nodes at transformer boundaries"""
        n_nodes = len(transformer_groups)
        boundary = torch.zeros(n_nodes, dtype=torch.bool)
        
        row, col = edge_index
        for i in range(edge_index.shape[1]):
            if transformer_groups[row[i]] != transformer_groups[col[i]]:
                boundary[row[i]] = True
                boundary[col[i]] = True
        
        return boundary
    
    def _calculate_cascade_potential_labels(
        self,
        features: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Calculate cascade potential for each node"""
        n_nodes = features.size(0)
        potential = torch.zeros(n_nodes)
        
        # Nodes with diverse neighbors have high cascade potential
        row, col = edge_index
        for i in range(n_nodes):
            neighbors = col[row == i]
            if len(neighbors) > 0:
                neighbor_features = features[neighbors]
                # Diversity as std of neighbor features
                diversity = neighbor_features.std(dim=0).mean()
                potential[i] = diversity
        
        return potential
    
    def _identify_complementarity_hubs(
        self,
        temporal_profiles: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Identify nodes that connect complementary profiles"""
        n_nodes = temporal_profiles.size(0)
        hub_score = torch.zeros(n_nodes)
        
        row, col = edge_index
        for i in range(n_nodes):
            neighbors = col[row == i]
            if len(neighbors) > 0:
                # Calculate complementarity with neighbors
                node_profile = temporal_profiles[i]
                neighbor_profiles = temporal_profiles[neighbors]
                
                # Correlation (negative = complementary)
                correlations = torch.tensor([
                    torch.corrcoef(torch.stack([node_profile, prof]))[0, 1]
                    for prof in neighbor_profiles
                ])
                
                # High score for negative correlations
                hub_score[i] = (-correlations).mean()
        
        return hub_score > hub_score.quantile(0.7)
    
    def _generate_solar_irradiance_profile(self) -> np.ndarray:
        """
        Generate realistic hourly solar irradiance profile for a day
        Returns values in W/m² 
        """
        import numpy as np
        hours = np.arange(24)
        
        # Simple bell curve for solar irradiance (peak at noon)
        # Max ~1000 W/m² at solar noon, 0 at night
        sunrise = 6
        sunset = 18
        peak_hour = 12
        
        irradiance = np.zeros(24)
        for h in range(24):
            if sunrise <= h <= sunset:
                # Sine curve for daylight hours
                angle = np.pi * (h - sunrise) / (sunset - sunrise)
                irradiance[h] = 1000 * np.sin(angle) * (0.8 + 0.2 * np.random.rand())  # Add some cloud variation
        
        return irradiance
    
    def _initialize_network_state(self, data: Data) -> Dict[str, torch.Tensor]:
        """Initialize network state tensors"""
        n_nodes = data.x.size(0)
        device = data.x.device  # Get device from data
        
        state = {
            'demand': data.temporal_profiles.mean(dim=1) if data.temporal_profiles is not None 
                     else torch.rand(n_nodes, device=device) * 10,
            'generation': torch.zeros(n_nodes, device=device),
            'net_demand': data.temporal_profiles.mean(dim=1) if data.temporal_profiles is not None 
                         else torch.rand(n_nodes, device=device) * 10,
            'congestion': torch.rand(n_nodes, device=device) * 0.5,  # Initial congestion
            'complementarity': torch.zeros(n_nodes, n_nodes, device=device)
        }
        
        # Calculate initial complementarity
        if data.temporal_profiles is not None:
            profiles_norm = (data.temporal_profiles - data.temporal_profiles.mean(dim=1, keepdim=True)) / (
                data.temporal_profiles.std(dim=1, keepdim=True) + 1e-8
            )
            state['complementarity'] = torch.matmul(profiles_norm, profiles_norm.t()) / data.temporal_profiles.size(1)
        
        return state
    
    def _update_features_with_state(
        self,
        features: torch.Tensor,
        network_state: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Update node features with current network state"""
        updated = features.clone()
        
        # Add network state information to features
        # This is simplified - in practice, concatenate or modify specific indices
        if 'generation' in network_state:
            # Assume last feature is generation capacity
            updated[:, -1] = network_state['generation'] / network_state['generation'].max() if network_state['generation'].max() > 0 else network_state['generation']
        
        return updated
    
    def _baseline_selection(self, data: Data, k: int = 10) -> List[int]:
        """Simple baseline selection by energy label and roof area"""
        # Score = poor energy label * roof area
        energy_scores = data.x[:, 0]  # Lower = worse label
        roof_scores = data.x[:, 2]  # Normalized roof area
        
        baseline_scores = (1 - energy_scores) * roof_scores
        
        # Select top k
        _, indices = torch.topk(baseline_scores, k)
        
        return indices.tolist()
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'round': self.intervention_round,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config
        }
        
        path = self.experiment_dir / filename
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.current_epoch = checkpoint['epoch']
        self.intervention_round = checkpoint['round']
        self.best_loss = checkpoint['best_loss']
        
        logger.info(f"Loaded checkpoint from {path}")
    
    def save_results(self, results: Dict[str, Any]):
        """Save experiment results"""
        # Save as JSON
        results_path = self.experiment_dir / 'results.json'
        
        # Convert tensors to lists for JSON serialization
        def tensor_to_list(obj):
            if isinstance(obj, torch.Tensor):
                return obj.cpu().tolist()
            elif isinstance(obj, dict):
                return {k: tensor_to_list(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [tensor_to_list(item) for item in obj]
            else:
                return obj
        
        serializable_results = tensor_to_list(results)
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Saved results to {results_path}")