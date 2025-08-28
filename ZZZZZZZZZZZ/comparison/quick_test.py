import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import generate_synthetic_data
from tier1_baselines.kmeans_clustering import KMeansMethod
from tier1_baselines.spectral_clustering import SpectralMethod
from tier1_baselines.louvain_clustering import LouvainMethod
from tier2_complementarity.correlation_clustering import CorrelationClusteringMethod
from tier2_complementarity.stable_matching import StableMatchingMethod

def quick_test():
    print("Generating synthetic data...")
    input_data = generate_synthetic_data(n_buildings=50)
    
    methods = {
        'K-means': KMeansMethod(n_clusters=5),
        'Spectral': SpectralMethod(n_clusters=5),
        'Louvain': LouvainMethod(),
        'Correlation': CorrelationClusteringMethod(n_clusters=5),
        'Stable Matching': StableMatchingMethod()
    }
    
    results = []
    for name, method in methods.items():
        print(f"Running {name}...")
        result = method.run(input_data)
        results.append({
            'Method': name,
            'Self-Sufficiency': result.metrics['self_sufficiency'],
            'Peak Reduction': result.metrics['peak_reduction'],
            'Violations': result.metrics['violations'],
            'Time (s)': result.computation_time
        })
    
    df = pd.DataFrame(results)
    df = df.sort_values('Self-Sufficiency', ascending=False)
    
    print("\nRESULTS:")
    print(df.to_string(index=False))
    
    df.to_csv('comparison/results/quick_test_results.csv', index=False)
    print("\nResults saved to comparison/results/quick_test_results.csv")

if __name__ == "__main__":
    quick_test()