import numpy as np
from typing import List, Tuple, Dict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.base_method import BaseClusteringMethod

class StableMatchingMethod(BaseClusteringMethod):
    def __init__(self, alpha=0.7, beta=0.3, max_group_size=10):
        super().__init__("Stable Matching with Complementarity")
        self.alpha = alpha
        self.beta = beta
        self.max_group_size = max_group_size
        self.matching_scores = None
        
    def fit_predict(self, input_data):
        consumption = input_data['consumption']
        generation = input_data.get('generation', np.zeros_like(consumption))
        
        n_buildings = consumption.shape[1]
        
        producers = []
        consumers = []
        for i in range(n_buildings):
            if generation[:, i].sum() > 0:
                producers.append(i)
            else:
                consumers.append(i)
        
        if len(producers) == 0:
            producers = list(range(0, n_buildings, 3))
            consumers = [i for i in range(n_buildings) if i not in producers]
        
        self.matching_scores = self._calculate_complementarity_matrix(
            consumption, generation, producers, consumers
        )
        
        matches = self._gale_shapley_many_to_many(
            producers, consumers, self.matching_scores
        )
        
        clusters = np.zeros(n_buildings, dtype=int)
        cluster_id = 0
        
        for producer, matched_consumers in matches.items():
            group = [producer] + list(matched_consumers)
            for building in group:
                clusters[building] = cluster_id
            cluster_id += 1
        
        unmatched = [i for i in range(n_buildings) if clusters[i] == 0 and i not in matches]
        if unmatched:
            for building in unmatched:
                clusters[building] = cluster_id
            cluster_id += 1
        
        return clusters
    
    def _calculate_complementarity_matrix(self, consumption, generation, producers, consumers):
        scores = {}
        
        for p in producers:
            scores[p] = {}
            p_gen = generation[:, p]
            
            for c in consumers:
                c_cons = consumption[:, c]
                
                overlap = np.minimum(p_gen, c_cons).sum()
                total_gen = p_gen.sum()
                complementarity = overlap / total_gen if total_gen > 0 else 0
                
                time_correlation = np.corrcoef(p_gen, c_cons)[0, 1]
                stability = 1 - abs(time_correlation) if time_correlation < 0 else 0
                
                scores[p][c] = self.alpha * complementarity + self.beta * stability
        
        return scores
    
    def _gale_shapley_many_to_many(self, producers, consumers, scores):
        max_matches_per_producer = min(self.max_group_size - 1, len(consumers) // max(len(producers), 1))
        
        producer_prefs = {}
        for p in producers:
            consumer_scores = [(c, scores[p][c]) for c in consumers]
            consumer_scores.sort(key=lambda x: x[1], reverse=True)
            producer_prefs[p] = [c for c, _ in consumer_scores]
        
        consumer_prefs = {}
        for c in consumers:
            producer_scores = [(p, scores[p][c]) for p in producers]
            producer_scores.sort(key=lambda x: x[1], reverse=True)
            consumer_prefs[c] = [p for p, _ in producer_scores]
        
        matches = {p: set() for p in producers}
        consumer_matched = {c: None for c in consumers}
        
        unmatched_producers = set(producers)
        
        while unmatched_producers:
            p = unmatched_producers.pop()
            
            for c in producer_prefs[p]:
                if len(matches[p]) >= max_matches_per_producer:
                    break
                    
                if consumer_matched[c] is None:
                    matches[p].add(c)
                    consumer_matched[c] = p
                else:
                    current_p = consumer_matched[c]
                    if consumer_prefs[c].index(p) < consumer_prefs[c].index(current_p):
                        matches[current_p].remove(c)
                        if len(matches[current_p]) < max_matches_per_producer:
                            unmatched_producers.add(current_p)
                        
                        matches[p].add(c)
                        consumer_matched[c] = p
            
            if len(matches[p]) < max_matches_per_producer and p not in unmatched_producers:
                unmatched_producers.add(p)
        
        return matches
    
    def get_additional_info(self):
        if self.matching_scores is not None:
            all_scores = []
            for p_scores in self.matching_scores.values():
                all_scores.extend(p_scores.values())
            
            return {
                'avg_matching_score': float(np.mean(all_scores)) if all_scores else 0,
                'max_matching_score': float(np.max(all_scores)) if all_scores else 0,
                'n_producers': len(self.matching_scores),
                'alpha': self.alpha,
                'beta': self.beta
            }
        return {}