"""
Method 2: Stable Matching with Prosumer-Consumer Pairing
Based on: "Stable Matching for Peer-to-Peer Energy Trading" 
(Morstyn et al., Applied Energy, 2018)

Implements Gale-Shapley algorithm with:
- Prosumers (has_solar=True OR has_battery=True) as proposers
- Consumers as acceptors
- Preference based on energy overlap and distance
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
import logging
from collections import defaultdict
from base_method import BaseClusteringMethod

logger = logging.getLogger(__name__)

class StableMatchingEnergySharing(BaseClusteringMethod):
    """
    Stable matching algorithm for prosumer-consumer pairing in energy sharing.
    Creates clusters based on stable bilateral agreements.
    """
    
    def __init__(self, alpha: float = 0.5, beta: float = 0.3, gamma: float = 0.2,
                 max_partners: int = 5):
        """
        Initialize Stable Matching for Energy Sharing.
        
        Args:
            alpha: Weight for energy overlap in utility function
            beta: Weight for electrical distance (inverse)
            gamma: Weight for price benefit
            max_partners: Maximum partners per prosumer (many-to-many matching)
        """
        super().__init__(
            name="Stable Matching Energy Sharing",
            paper_reference="Morstyn et al., Applied Energy, 2018"
        )
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.max_partners = max_partners
        
        self.prosumers = []
        self.consumers = []
        self.preferences = {}
        self.matches = {}
        
        logger.info(f"Initialized with α={alpha}, β={beta}, γ={gamma}, max_partners={max_partners}")
    
    def _perform_clustering(self, **kwargs) -> Dict[str, List[str]]:
        """
        Perform stable matching clustering.
        
        Returns:
            Dictionary mapping cluster_id -> list of building_ids
        """
        # Update parameters if provided
        self.alpha = kwargs.get('alpha', self.alpha)
        self.beta = kwargs.get('beta', self.beta)
        self.gamma = kwargs.get('gamma', self.gamma)
        self.max_partners = kwargs.get('max_partners', self.max_partners)
        
        # Identify prosumers and consumers
        self._identify_agents()
        
        # Calculate preferences
        self.preferences = self._calculate_preferences()
        
        # Run Gale-Shapley algorithm
        self.matches = self._gale_shapley_with_capacity()
        
        # Convert matches to clusters
        clusters = self._matches_to_clusters()
        
        return clusters
    
    def _identify_agents(self):
        """
        Identify prosumers and consumers from building features.
        From paper's Section 3.2: Prosumers have generation capability.
        """
        logger.info("Identifying prosumers and consumers...")
        
        building_features = self.preprocessed_data['building_features']
        
        # Prosumers: buildings with solar, battery, or high solar potential
        prosumer_mask = (
            (building_features['has_solar'] == True) |
            (building_features['has_battery'] == True) |
            (building_features['solar_potential'].isin(['high', 'medium']))
        )
        
        self.prosumers = building_features[prosumer_mask]['ogc_fid'].tolist()
        self.consumers = building_features[~prosumer_mask]['ogc_fid'].tolist()
        
        logger.info(f"Identified {len(self.prosumers)} prosumers and {len(self.consumers)} consumers")
        
        # If too few prosumers, convert some high-potential consumers
        if len(self.prosumers) < len(self.consumers) / 10:
            logger.warning("Too few prosumers, converting high-potential consumers...")
            
            # Convert top 10% of consumers with best solar potential
            potential_prosumers = building_features[
                (~prosumer_mask) & 
                (building_features['suitable_roof_area'] > 50)
            ].nlargest(len(self.consumers) // 10, 'suitable_roof_area')
            
            for bid in potential_prosumers['ogc_fid']:
                if bid in self.consumers:
                    self.consumers.remove(bid)
                    self.prosumers.append(bid)
            
            logger.info(f"After conversion: {len(self.prosumers)} prosumers")
    
    def _calculate_preferences(self) -> Dict[str, List[Tuple[str, float]]]:
        """
        Calculate preferences based on utility function.
        Paper's Equation (8):
        preference_score = Σ_t min(generation_i(t), consumption_j(t)) / Σ_t consumption_j(t)
        
        Modified for KG constraints:
        - Set preference to 0 if different CableGroup
        - Multiply by 0.5 if different Transformer
        """
        logger.info("Calculating preference scores...")
        
        preferences = {
            'prosumer': defaultdict(list),
            'consumer': defaultdict(list)
        }
        
        time_series = self.preprocessed_data['time_series']
        constraints = self.preprocessed_data['constraints']
        electrical_distances = self.preprocessed_data['electrical_distances']
        bid_to_idx = constraints['bid_to_idx']
        
        # Calculate utility for each prosumer-consumer pair
        for prosumer in self.prosumers:
            prosumer_utilities = []
            
            for consumer in self.consumers:
                utility = self._calculate_utility(
                    prosumer, consumer, 
                    time_series, constraints, 
                    electrical_distances, bid_to_idx
                )
                prosumer_utilities.append((consumer, utility))
            
            # Sort by utility (descending)
            prosumer_utilities.sort(key=lambda x: x[1], reverse=True)
            preferences['prosumer'][prosumer] = prosumer_utilities
        
        # Calculate consumer preferences (inverse perspective)
        for consumer in self.consumers:
            consumer_utilities = []
            
            for prosumer in self.prosumers:
                utility = self._calculate_utility(
                    prosumer, consumer,
                    time_series, constraints,
                    electrical_distances, bid_to_idx
                )
                consumer_utilities.append((prosumer, utility))
            
            # Sort by utility (descending)
            consumer_utilities.sort(key=lambda x: x[1], reverse=True)
            preferences['consumer'][consumer] = consumer_utilities
        
        logger.info("Preference calculation complete")
        
        return preferences
    
    def _calculate_utility(self, prosumer: str, consumer: str,
                          time_series: Dict, constraints: Dict,
                          electrical_distances: np.ndarray,
                          bid_to_idx: Dict) -> float:
        """
        Calculate utility for a prosumer-consumer pair.
        
        Utility function: U_ij = α * energy_overlap + β * (1/electrical_distance) + γ * price_benefit
        """
        # Check if both buildings are in the index
        if prosumer not in bid_to_idx or consumer not in bid_to_idx:
            return 0.0
        
        p_idx = bid_to_idx[prosumer]
        c_idx = bid_to_idx[consumer]
        
        # Check cable group constraint
        same_cable_group = constraints['same_cable_group'][p_idx, c_idx]
        same_transformer = constraints['same_transformer'][p_idx, c_idx]
        
        if not same_cable_group and not same_transformer:
            return 0.0  # Cannot trade across different transformers
        
        # Calculate energy overlap
        energy_overlap = self._calculate_energy_overlap(prosumer, consumer, time_series)
        
        # Calculate distance factor
        elec_distance = electrical_distances[p_idx, c_idx]
        distance_factor = 1.0 / (1 + elec_distance) if elec_distance > 0 else 1.0
        
        # Calculate price benefit (simplified: based on complementarity)
        complementarity = self.preprocessed_data['complementarity'][p_idx, c_idx]
        price_benefit = complementarity  # Higher complementarity = better price
        
        # Apply constraint multipliers
        constraint_multiplier = 1.0 if same_cable_group else 0.5
        
        # Calculate total utility
        utility = constraint_multiplier * (
            self.alpha * energy_overlap +
            self.beta * distance_factor +
            self.gamma * price_benefit
        )
        
        return utility
    
    def _calculate_energy_overlap(self, prosumer: str, consumer: str,
                                 time_series: Dict) -> float:
        """
        Calculate energy overlap between prosumer generation and consumer demand.
        Paper's Equation (8): Σ_t min(generation_i(t), consumption_j(t)) / Σ_t consumption_j(t)
        """
        if prosumer not in time_series or consumer not in time_series:
            return 0.0
        
        p_ts = time_series[prosumer]
        c_ts = time_series[consumer]
        
        if len(p_ts) == 0 or len(c_ts) == 0:
            return 0.0
        
        # Get generation from prosumer (solar - column 5)
        p_generation = p_ts[:, 5] if p_ts.shape[1] > 5 else np.zeros(len(p_ts))
        
        # Get consumption from consumer (electricity demand - column 3)
        c_consumption = c_ts[:, 3] if c_ts.shape[1] > 3 else np.zeros(len(c_ts))
        
        # Calculate overlap
        overlap_sum = 0
        consumption_sum = 0
        
        for t in range(min(len(p_generation), len(c_consumption))):
            overlap_sum += min(p_generation[t], c_consumption[t])
            consumption_sum += c_consumption[t]
        
        if consumption_sum > 0:
            return overlap_sum / consumption_sum
        
        return 0.0
    
    def _gale_shapley_with_capacity(self) -> Dict[str, List[str]]:
        """
        Modified Gale-Shapley algorithm with capacity constraints.
        Algorithm 2 from paper with modifications:
        - Check transformer capacity before accepting match
        - Allow many-to-many matching within limits
        - Include battery storage in prosumer capacity
        """
        logger.info("Running Gale-Shapley algorithm with capacity constraints...")
        
        # Initialize data structures
        prosumer_partners = defaultdict(list)
        consumer_partners = defaultdict(list)
        prosumer_proposals = defaultdict(int)  # Track proposal index
        
        # Track unmatched prosumers
        unmatched = set(self.prosumers)
        
        # Maximum iterations to prevent infinite loops
        max_iterations = len(self.prosumers) * len(self.consumers)
        iteration = 0
        
        while unmatched and iteration < max_iterations:
            iteration += 1
            
            # Copy to avoid modification during iteration
            current_unmatched = list(unmatched)
            
            for prosumer in current_unmatched:
                # Check if prosumer has capacity for more partners
                if len(prosumer_partners[prosumer]) >= self.max_partners:
                    unmatched.discard(prosumer)
                    continue
                
                # Get next consumer to propose to
                pref_list = self.preferences['prosumer'].get(prosumer, [])
                proposal_idx = prosumer_proposals[prosumer]
                
                if proposal_idx >= len(pref_list):
                    # No more consumers to propose to
                    unmatched.discard(prosumer)
                    continue
                
                consumer, utility = pref_list[proposal_idx]
                prosumer_proposals[prosumer] += 1
                
                # Check if consumer accepts proposal
                if self._consumer_accepts(consumer, prosumer, consumer_partners[consumer]):
                    # Check transformer capacity
                    if self._check_capacity_for_match(prosumer, consumer):
                        # Accept match
                        prosumer_partners[prosumer].append(consumer)
                        consumer_partners[consumer].append(prosumer)
                        
                        # If consumer is at capacity, remove least preferred partner
                        if len(consumer_partners[consumer]) > self.max_partners:
                            # Find least preferred partner
                            consumer_prefs = self.preferences['consumer'].get(consumer, [])
                            pref_dict = {p: i for i, (p, _) in enumerate(consumer_prefs)}
                            
                            least_preferred = max(
                                consumer_partners[consumer],
                                key=lambda p: pref_dict.get(p, float('inf'))
                            )
                            
                            # Remove least preferred
                            consumer_partners[consumer].remove(least_preferred)
                            prosumer_partners[least_preferred].remove(consumer)
                            
                            # Least preferred becomes unmatched again
                            if len(prosumer_partners[least_preferred]) < self.max_partners:
                                unmatched.add(least_preferred)
                        
                        # Check if prosumer still has capacity
                        if len(prosumer_partners[prosumer]) < self.max_partners:
                            # Keep in unmatched to find more partners
                            pass
                        else:
                            unmatched.discard(prosumer)
        
        logger.info(f"Matching complete after {iteration} iterations")
        logger.info(f"Matched {len(prosumer_partners)} prosumers with "
                   f"{len(consumer_partners)} consumers")
        
        # Combine into final matches
        matches = {}
        matches['prosumer_partners'] = dict(prosumer_partners)
        matches['consumer_partners'] = dict(consumer_partners)
        
        return matches
    
    def _consumer_accepts(self, consumer: str, prosumer: str,
                         current_partners: List[str]) -> bool:
        """
        Check if consumer accepts prosumer's proposal.
        """
        # Always accept if consumer has capacity
        if len(current_partners) < self.max_partners:
            return True
        
        # If at capacity, check if prosumer is preferred over current partners
        consumer_prefs = self.preferences['consumer'].get(consumer, [])
        pref_dict = {p: i for i, (p, _) in enumerate(consumer_prefs)}
        
        # Get preference rank of prosumer
        prosumer_rank = pref_dict.get(prosumer, float('inf'))
        
        # Check if prosumer is preferred over any current partner
        for partner in current_partners:
            partner_rank = pref_dict.get(partner, float('inf'))
            if prosumer_rank < partner_rank:
                return True
        
        return False
    
    def _check_capacity_for_match(self, prosumer: str, consumer: str) -> bool:
        """
        Check if match respects transformer capacity constraints.
        """
        constraints = self.preprocessed_data['constraints']
        
        # Find transformer for this pair
        for t_id, t_buildings in constraints['transformer_groups'].items():
            if prosumer in [str(b) for b in t_buildings] and \
               consumer in [str(b) for b in t_buildings]:
                # Both in same transformer group
                return self._check_transformer_capacity([prosumer, consumer], t_id)
        
        return True  # If not in same transformer, already filtered in utility
    
    def _matches_to_clusters(self) -> Dict[str, List[str]]:
        """
        Convert stable matches to clusters.
        Each match becomes a small cluster.
        """
        logger.info("Converting matches to clusters...")
        
        clusters = {}
        cluster_id = 0
        
        # Create clusters from prosumer partnerships
        for prosumer, consumers in self.matches.get('prosumer_partners', {}).items():
            if consumers:
                # Create cluster with prosumer and matched consumers
                cluster_members = [prosumer] + consumers
                clusters[f"stable_{cluster_id}"] = cluster_members
                cluster_id += 1
        
        # Add unmatched buildings as singleton clusters or merge nearby ones
        all_matched = set()
        for prosumer, consumers in self.matches.get('prosumer_partners', {}).items():
            all_matched.add(prosumer)
            all_matched.update(consumers)
        
        # Find unmatched buildings
        all_buildings = set(self.preprocessed_data['building_features']['ogc_fid'])
        unmatched = all_buildings - all_matched
        
        if unmatched:
            logger.info(f"Handling {len(unmatched)} unmatched buildings...")
            
            # Group unmatched by cable group
            constraints = self.preprocessed_data['constraints']
            cable_group_unmatched = defaultdict(list)
            
            for bid in unmatched:
                if bid in constraints['bid_to_idx']:
                    idx = constraints['bid_to_idx'][bid]
                    for cg_id, cg_indices in constraints['cable_groups'].items():
                        if idx in cg_indices:
                            cable_group_unmatched[cg_id].append(bid)
                            break
            
            # Create clusters for unmatched buildings in same cable group
            for cg_id, buildings in cable_group_unmatched.items():
                if len(buildings) >= 3:  # Minimum cluster size
                    clusters[f"unmatched_{cg_id}"] = buildings
                    cluster_id += 1
        
        logger.info(f"Created {len(clusters)} clusters from stable matching")
        
        return clusters
    
    def get_method_specific_metrics(self) -> Dict[str, Any]:
        """
        Get metrics specific to stable matching.
        """
        if not self.matches:
            return {}
        
        prosumer_partners = self.matches.get('prosumer_partners', {})
        consumer_partners = self.matches.get('consumer_partners', {})
        
        # Calculate matching statistics
        prosumer_match_counts = [len(partners) for partners in prosumer_partners.values()]
        consumer_match_counts = [len(partners) for partners in consumer_partners.values()]
        
        metrics = {
            'prosumer_utilization': len(prosumer_partners) / len(self.prosumers) if self.prosumers else 0,
            'consumer_satisfaction': len(consumer_partners) / len(self.consumers) if self.consumers else 0,
            'avg_partners_per_prosumer': np.mean(prosumer_match_counts) if prosumer_match_counts else 0,
            'avg_partners_per_consumer': np.mean(consumer_match_counts) if consumer_match_counts else 0,
            'total_matches': sum(prosumer_match_counts),
            'stability': self._check_stability()
        }
        
        return metrics
    
    def _check_stability(self) -> bool:
        """
        Check if matching is stable (no blocking pairs).
        A blocking pair exists if both prefer each other over current matches.
        """
        if not self.matches or not self.preferences:
            return True
        
        prosumer_partners = self.matches.get('prosumer_partners', {})
        consumer_partners = self.matches.get('consumer_partners', {})
        
        # Check for blocking pairs
        for prosumer in self.prosumers:
            for consumer in self.consumers:
                # Check if they're not matched
                if consumer not in prosumer_partners.get(prosumer, []):
                    # Check if they would prefer each other
                    
                    # Get prosumer's preference for consumer
                    p_prefs = self.preferences['prosumer'].get(prosumer, [])
                    p_pref_dict = {c: utility for c, utility in p_prefs}
                    p_utility_new = p_pref_dict.get(consumer, 0)
                    
                    # Get prosumer's worst current partner utility
                    p_worst_current = float('inf')
                    for current_c in prosumer_partners.get(prosumer, []):
                        p_worst_current = min(p_worst_current, p_pref_dict.get(current_c, 0))
                    
                    # Get consumer's preference for prosumer
                    c_prefs = self.preferences['consumer'].get(consumer, [])
                    c_pref_dict = {p: utility for p, utility in c_prefs}
                    c_utility_new = c_pref_dict.get(prosumer, 0)
                    
                    # Get consumer's worst current partner utility
                    c_worst_current = float('inf')
                    for current_p in consumer_partners.get(consumer, []):
                        c_worst_current = min(c_worst_current, c_pref_dict.get(current_p, 0))
                    
                    # Check if this would be a blocking pair
                    if p_utility_new > p_worst_current and c_utility_new > c_worst_current:
                        return False  # Found blocking pair
        
        return True  # No blocking pairs found