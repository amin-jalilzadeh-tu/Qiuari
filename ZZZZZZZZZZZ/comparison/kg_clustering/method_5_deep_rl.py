"""
Method 5: Deep Reinforcement Learning Clustering
Based on: "Deep Reinforcement Learning for Optimal Energy Community Formation" 
(Zhang et al., Applied Energy, 2023)

Use Graph Neural Network with RL for clustering:
1. State: Current clustering + KG features
2. Action: Move building to different cluster
3. Reward: Peak reduction + self-sufficiency + constraint satisfaction
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
from typing import Dict, List, Any, Tuple, Optional
import logging
from collections import defaultdict
import random
from base_method import BaseClusteringMethod

logger = logging.getLogger(__name__)

class GNNEncoder(nn.Module):
    """
    Graph Neural Network encoder for state representation.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 32):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x, edge_index, batch=None):
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        return x

class PolicyNetwork(nn.Module):
    """
    Policy network for action selection.
    """
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return F.softmax(logits, dim=-1)

class ValueNetwork(nn.Module):
    """
    Value network for state evaluation.
    """
    def __init__(self, state_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value

class RLClusteringWithKG(BaseClusteringMethod):
    """
    Deep RL clustering using Graph Neural Networks.
    Learns optimal clustering policy through interaction with environment.
    """
    
    def __init__(self, n_episodes: int = 100, max_steps: int = 200,
                 learning_rate: float = 0.001, gamma: float = 0.99,
                 epsilon: float = 0.1, alpha: float = 1.0, 
                 beta: float = 1.0, mu: float = 10.0):
        """
        Initialize RL Clustering with KG.
        
        Args:
            n_episodes: Number of training episodes
            max_steps: Maximum steps per episode
            learning_rate: Learning rate for neural networks
            gamma: Discount factor
            epsilon: Exploration rate
            alpha: Weight for peak reduction in reward
            beta: Weight for self-sufficiency in reward
            mu: Weight for constraint penalty
        """
        super().__init__(
            name="Deep RL Clustering with KG",
            paper_reference="Zhang et al., Applied Energy, 2023"
        )
        
        self.n_episodes = n_episodes
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Networks will be initialized when data is available
        self.gnn_encoder = None
        self.policy_net = None
        self.value_net = None
        
        # Training history
        self.episode_rewards = []
        self.best_clustering = None
        self.best_reward = -float('inf')
        
        logger.info(f"Initialized RL clustering on device: {self.device}")
    
    def _perform_clustering(self, **kwargs) -> Dict[str, List[str]]:
        """
        Perform RL-based clustering.
        
        Returns:
            Dictionary mapping cluster_id -> list of building_ids
        """
        # Update parameters if provided
        self.n_episodes = kwargs.get('n_episodes', self.n_episodes)
        self.epsilon = kwargs.get('epsilon', self.epsilon)
        
        # Initialize environment and networks
        self._initialize_environment()
        self._initialize_networks()
        
        # Train RL agent
        self._train_agent()
        
        # Return best clustering found
        return self.best_clustering if self.best_clustering else {}
    
    def _initialize_environment(self):
        """
        Initialize RL environment from KG data.
        """
        logger.info("Initializing RL environment from KG...")
        
        # Extract features
        self.building_features = self.preprocessed_data['building_features']
        self.complementarity = self.preprocessed_data['complementarity']
        self.constraints = self.preprocessed_data['constraints']
        self.time_series = self.preprocessed_data['time_series']
        
        # Create initial random clustering
        self.n_buildings = len(self.building_features)
        self.n_clusters = min(20, self.n_buildings // 5)  # Adaptive cluster count
        
        # Building indices
        self.building_ids = self.building_features['ogc_fid'].tolist()
        self.bid_to_idx = {bid: i for i, bid in enumerate(self.building_ids)}
        
        logger.info(f"Environment initialized with {self.n_buildings} buildings")
    
    def _initialize_networks(self):
        """
        Initialize neural networks for RL.
        """
        logger.info("Initializing neural networks...")
        
        # Feature dimensions
        node_features = self._extract_node_features()
        input_dim = node_features.shape[1]
        state_dim = 32  # GNN output dimension
        action_dim = self.n_clusters + 1  # Cluster assignment + no-op
        
        # Initialize networks
        self.gnn_encoder = GNNEncoder(input_dim, 64, state_dim).to(self.device)
        self.policy_net = PolicyNetwork(state_dim * 2, action_dim).to(self.device)  # *2 for node + graph features
        self.value_net = ValueNetwork(state_dim * 2).to(self.device)
        
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=self.learning_rate)
        self.gnn_optimizer = torch.optim.Adam(self.gnn_encoder.parameters(), lr=self.learning_rate)
        
        logger.info("Networks initialized")
    
    def _extract_node_features(self) -> np.ndarray:
        """
        Extract node features from KG for GNN.
        """
        features = []
        
        for _, row in self.building_features.iterrows():
            node_feat = [
                row['area'] / 1000,  # Normalize
                row['height'] / 20,
                row['energy_label_num'] / 7,
                float(row['has_solar']),
                float(row['has_battery']),
                float(row['has_heat_pump']),
                row['solar_capacity_kwp'] / 50,
                row['suitable_roof_area'] / 200,
                float(row['is_residential']),
                row['expected_cop'] / 5
            ]
            features.append(node_feat)
        
        return np.array(features, dtype=np.float32)
    
    def _create_state_representation(self, clustering: Dict[int, List[int]]) -> torch.Tensor:
        """
        Create state representation from current clustering.
        Paper's state encoding (Section 4.3):
        - Node features from KG
        - Edge features (electrical distance, complementarity)
        - Current cluster assignments
        - Temporal features from EnergyState
        """
        # Node features
        node_features = self._extract_node_features()
        
        # Add cluster assignment as feature
        cluster_assignment = np.zeros((self.n_buildings, self.n_clusters))
        for cluster_id, members in clustering.items():
            for building_idx in members:
                cluster_assignment[building_idx, cluster_id] = 1
        
        # Combine features
        combined_features = np.concatenate([node_features, cluster_assignment], axis=1)
        x = torch.tensor(combined_features, dtype=torch.float32).to(self.device)
        
        # Create edge index from complementarity graph
        edge_list = []
        threshold = np.percentile(self.complementarity[self.complementarity > 0], 75)
        
        for i in range(self.n_buildings):
            for j in range(i + 1, self.n_buildings):
                if self.complementarity[i, j] > threshold:
                    edge_list.append([i, j])
                    edge_list.append([j, i])  # Undirected
        
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().to(self.device)
        else:
            # Fallback: create minimal connected graph
            edge_list = [[i, i+1] for i in range(self.n_buildings-1)]
            edge_list += [[i+1, i] for i in range(self.n_buildings-1)]
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().to(self.device)
        
        # Create PyG data object
        data = Data(x=x, edge_index=edge_index)
        
        # Get GNN encoding
        with torch.no_grad():
            node_embeddings = self.gnn_encoder(data.x, data.edge_index)
            graph_embedding = torch.mean(node_embeddings, dim=0, keepdim=True)
        
        return node_embeddings, graph_embedding
    
    def _calculate_reward(self, clustering: Dict[int, List[int]], action: Tuple[int, int]) -> float:
        """
        Calculate reward for current state and action.
        Equation (18) from paper:
        R = Σ_c [peak_before - peak_after] / peak_before +
            λ * diversity_index +
            μ * constraint_penalty
        """
        # Calculate peak reduction
        peak_reduction = self._calculate_peak_reduction_rl(clustering)
        
        # Calculate self-sufficiency
        self_sufficiency = self._calculate_self_sufficiency_rl(clustering)
        
        # Calculate diversity
        diversity = self._calculate_diversity_rl(clustering)
        
        # Calculate constraint violations
        violations = self._count_violations_rl(clustering)
        
        # Combine into reward
        reward = (self.alpha * peak_reduction + 
                 self.beta * self_sufficiency + 
                 0.5 * diversity - 
                 self.mu * violations)
        
        return reward
    
    def _calculate_peak_reduction_rl(self, clustering: Dict[int, List[int]]) -> float:
        """Calculate peak reduction for RL reward."""
        if not clustering:
            return 0.0
        
        total_individual = 0
        total_aggregated = 0
        
        for cluster_id, members in clustering.items():
            if not members:
                continue
            
            cluster_individual = 0
            cluster_aggregated = None
            
            for idx in members:
                bid = self.building_ids[idx]
                if bid in self.time_series:
                    ts = self.time_series[bid]
                    if len(ts) > 0:
                        demand = ts[:, 3]
                        cluster_individual += np.max(demand)
                        
                        if cluster_aggregated is None:
                            cluster_aggregated = demand.copy()
                        else:
                            cluster_aggregated += demand
            
            if cluster_aggregated is not None:
                total_individual += cluster_individual
                total_aggregated += np.max(cluster_aggregated)
        
        if total_individual > 0:
            return 1 - (total_aggregated / total_individual)
        return 0.0
    
    def _calculate_self_sufficiency_rl(self, clustering: Dict[int, List[int]]) -> float:
        """Calculate self-sufficiency for RL reward."""
        if not clustering:
            return 0.0
        
        ratios = []
        
        for cluster_id, members in clustering.items():
            if not members:
                continue
            
            generation = 0
            consumption = 0
            
            for idx in members:
                bid = self.building_ids[idx]
                if bid in self.time_series:
                    ts = self.time_series[bid]
                    if len(ts) > 0:
                        generation += np.sum(ts[:, 5])  # Solar
                        consumption += np.sum(ts[:, 3])  # Demand
            
            if consumption > 0:
                ratios.append(min(1.0, generation / consumption))
        
        return np.mean(ratios) if ratios else 0.0
    
    def _calculate_diversity_rl(self, clustering: Dict[int, List[int]]) -> float:
        """Calculate diversity index for RL reward."""
        if not clustering:
            return 0.0
        
        diversities = []
        
        for cluster_id, members in clustering.items():
            if len(members) < 2:
                continue
            
            # Get building types
            types = set()
            has_solar = 0
            has_battery = 0
            
            for idx in members:
                row = self.building_features.iloc[idx]
                types.add(row['building_function'])
                if row['has_solar']:
                    has_solar += 1
                if row['has_battery']:
                    has_battery += 1
            
            # Calculate diversity
            type_diversity = len(types) / max(3, len(types))
            asset_diversity = min(1.0, (has_solar + has_battery) / len(members))
            
            diversities.append((type_diversity + asset_diversity) / 2)
        
        return np.mean(diversities) if diversities else 0.0
    
    def _count_violations_rl(self, clustering: Dict[int, List[int]]) -> int:
        """Count constraint violations for RL penalty."""
        violations = 0
        
        for cluster_id, members in clustering.items():
            if len(members) < 2:
                continue
            
            # Check cable group violations
            cable_groups = set()
            for idx in members:
                for cg_id, cg_indices in self.constraints['cable_groups'].items():
                    if idx in cg_indices:
                        cable_groups.add(cg_id)
                        break
            
            if len(cable_groups) > 1:
                violations += len(cable_groups) - 1
        
        return violations
    
    def _train_agent(self):
        """
        Train RL agent through episodes.
        """
        logger.info(f"Training RL agent for {self.n_episodes} episodes...")
        
        for episode in range(self.n_episodes):
            # Initialize episode
            clustering = self._initialize_clustering()
            episode_reward = 0
            
            for step in range(self.max_steps):
                # Get state representation
                node_embeddings, graph_embedding = self._create_state_representation(clustering)
                
                # Select action (epsilon-greedy)
                if random.random() < self.epsilon:
                    # Random action
                    building_idx = random.randint(0, self.n_buildings - 1)
                    new_cluster = random.randint(0, self.n_clusters - 1)
                else:
                    # Policy action
                    building_idx = random.randint(0, self.n_buildings - 1)  # Simplified
                    
                    # Get building's state
                    building_state = torch.cat([
                        node_embeddings[building_idx],
                        graph_embedding.squeeze()
                    ])
                    
                    action_probs = self.policy_net(building_state.unsqueeze(0))
                    new_cluster = torch.argmax(action_probs).item()
                    
                    if new_cluster >= self.n_clusters:
                        continue  # No-op action
                
                # Execute action
                old_cluster = None
                for c_id, members in clustering.items():
                    if building_idx in members:
                        old_cluster = c_id
                        members.remove(building_idx)
                        break
                
                if new_cluster not in clustering:
                    clustering[new_cluster] = []
                clustering[new_cluster].append(building_idx)
                
                # Calculate reward
                reward = self._calculate_reward(clustering, (building_idx, new_cluster))
                episode_reward += reward
                
                # Update networks (simplified PPO update)
                if not random.random() < self.epsilon:
                    # Calculate advantage
                    state_value = self.value_net(building_state.unsqueeze(0))
                    next_node_embeddings, next_graph_embedding = self._create_state_representation(clustering)
                    next_state = torch.cat([
                        next_node_embeddings[building_idx],
                        next_graph_embedding.squeeze()
                    ])
                    next_value = self.value_net(next_state.unsqueeze(0))
                    
                    advantage = reward + self.gamma * next_value - state_value
                    
                    # Update policy
                    self.policy_optimizer.zero_grad()
                    action_probs = self.policy_net(building_state.unsqueeze(0))
                    policy_loss = -torch.log(action_probs[0, new_cluster]) * advantage.detach()
                    policy_loss.backward()
                    self.policy_optimizer.step()
                    
                    # Update value
                    self.value_optimizer.zero_grad()
                    value_loss = F.mse_loss(state_value, reward + self.gamma * next_value.detach())
                    value_loss.backward()
                    self.value_optimizer.step()
            
            # Store episode results
            self.episode_rewards.append(episode_reward)
            
            # Update best clustering
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                self.best_clustering = self._convert_clustering_format(clustering)
            
            # Decay epsilon
            self.epsilon *= 0.995
            
            if episode % 10 == 0:
                logger.info(f"Episode {episode}: Reward = {episode_reward:.3f}, "
                           f"Best = {self.best_reward:.3f}, ε = {self.epsilon:.3f}")
        
        logger.info(f"Training complete. Best reward: {self.best_reward:.3f}")
    
    def _initialize_clustering(self) -> Dict[int, List[int]]:
        """Initialize random valid clustering."""
        clustering = defaultdict(list)
        
        # Assign buildings to clusters respecting cable groups
        for i, bid in enumerate(self.building_ids):
            # Find cable group
            cg_id = None
            for cg, indices in self.constraints['cable_groups'].items():
                if i in indices:
                    cg_id = cg
                    break
            
            # Assign to cluster based on cable group
            if cg_id:
                cluster_id = hash(cg_id) % self.n_clusters
            else:
                cluster_id = random.randint(0, self.n_clusters - 1)
            
            clustering[cluster_id].append(i)
        
        return dict(clustering)
    
    def _convert_clustering_format(self, clustering: Dict[int, List[int]]) -> Dict[str, List[str]]:
        """Convert internal clustering format to output format."""
        output_clusters = {}
        
        for cluster_id, member_indices in clustering.items():
            if member_indices and len(member_indices) >= 3:
                building_ids = [self.building_ids[idx] for idx in member_indices]
                output_clusters[f"rl_{cluster_id}"] = building_ids
        
        return output_clusters
    
    def get_method_specific_metrics(self) -> Dict[str, Any]:
        """Get RL-specific metrics."""
        if not self.episode_rewards:
            return {}
        
        return {
            'final_reward': self.best_reward,
            'avg_episode_reward': np.mean(self.episode_rewards),
            'reward_improvement': self.episode_rewards[-1] - self.episode_rewards[0] if len(self.episode_rewards) > 1 else 0,
            'convergence_episode': np.argmax(self.episode_rewards) if self.episode_rewards else 0,
            'final_epsilon': self.epsilon
        }