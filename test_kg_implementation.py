"""
Comprehensive Knowledge Graph Implementation Validation
Tests KG connector design, schema, and query patterns
"""

import sys
import ast
import re
from typing import Dict, List, Tuple, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KGImplementationValidator:
    """Validates the knowledge graph implementation without live connection."""
    
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.strengths = []
        self.metrics = {}
        
    def analyze_kg_connector(self):
        """Analyze the kg_connector.py implementation."""
        logger.info("Analyzing KG Connector Implementation...")
        
        # Read the kg_connector source
        with open('data/kg_connector.py', 'r') as f:
            source = f.read()
            
        # Parse AST for structural analysis
        tree = ast.parse(source)
        
        # Extract class and method information
        classes = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes[node.name] = {
                    'methods': [],
                    'docstring': ast.get_docstring(node),
                    'line': node.lineno
                }
                for item in node.body:
                    if isinstance(node, ast.FunctionDef):
                        classes[node.name]['methods'].append(item.name)
        
        # Analyze schema design
        self._analyze_schema_design(source)
        
        # Analyze Cypher queries
        self._analyze_cypher_queries(source)
        
        # Check error handling
        self._check_error_handling(source)
        
        # Validate data pipeline integration
        self._validate_pipeline_integration(source)
        
        # Check performance optimizations
        self._check_performance_patterns(source)
        
    def _analyze_schema_design(self, source: str):
        """Analyze the graph schema design."""
        logger.info("  Validating schema design...")
        
        # Expected node types for energy system
        expected_nodes = [
            'Building', 'CableGroup', 'Transformer', 'Substation',
            'AdjacencyCluster', 'EnergyState', 'TimeSlot',
            'SolarSystem', 'BatterySystem', 'HeatPumpSystem'
        ]
        
        # Expected relationships
        expected_rels = [
            'CONNECTED_TO', 'CONNECTS_TO', 'FEEDS_FROM',
            'IN_ADJACENCY_CLUSTER', 'FOR_BUILDING', 'DURING',
            'HAS_INSTALLED'
        ]
        
        # Check node presence in queries
        for node in expected_nodes:
            if f':{node}' in source:
                self.strengths.append(f"[OK] {node} node type implemented")
            else:
                self.warnings.append(f"[WARN] {node} node type not found in queries")
        
        # Check relationship presence
        for rel in expected_rels:
            if f':{rel}' in source or f'-[:{rel}]-' in source:
                self.strengths.append(f"[OK] {rel} relationship implemented")
            else:
                self.warnings.append(f"[WARN] {rel} relationship not found")
        
        # Check for proper hierarchy
        hierarchy_query = 'Building.*CONNECTED_TO.*CableGroup.*CONNECTS_TO.*Transformer.*FEEDS_FROM.*Substation'
        if re.search(hierarchy_query, source, re.DOTALL):
            self.strengths.append("[OK] Proper grid hierarchy maintained")
        else:
            self.issues.append("[ERROR] Grid hierarchy may be incomplete")
            
    def _analyze_cypher_queries(self, source: str):
        """Analyze Cypher query patterns and quality."""
        logger.info("  Analyzing Cypher queries...")
        
        # Extract all Cypher queries
        cypher_pattern = r'"""[\s\S]*?(?:MATCH|CREATE|MERGE|RETURN)[\s\S]*?"""'
        queries = re.findall(cypher_pattern, source)
        
        self.metrics['total_queries'] = len(queries)
        logger.info(f"    Found {len(queries)} Cypher queries")
        
        query_issues = []
        
        for i, query in enumerate(queries):
            # Check for common anti-patterns
            
            # 1. Missing WHERE clause optimization
            if 'MATCH' in query and 'WHERE' not in query and 'OPTIONAL MATCH' not in query:
                query_issues.append(f"Query {i+1}: No WHERE clause for filtering")
            
            # 2. Check for proper parameter binding
            if '$' in query:
                self.strengths.append(f"[OK] Query {i+1} uses parameter binding")
            elif 'MATCH' in query and not any(x in query for x in ['RETURN 1', 'CALL db']):
                query_issues.append(f"Query {i+1}: No parameter binding (potential injection risk)")
            
            # 3. Check for index hints
            if 'Building' in query and 'district_name' in query:
                if 'district_name: $' in query or '{district_name:' in query:
                    self.strengths.append(f"[OK] Query {i+1} filters on indexed property")
            
            # 4. Check for OPTIONAL MATCH usage
            if 'OPTIONAL MATCH' in query:
                self.strengths.append(f"[OK] Query {i+1} handles optional relationships")
            
            # 5. Check for proper aggregation
            if any(agg in query for agg in ['count(', 'sum(', 'avg(', 'collect(']):
                if 'WITH' in query or 'RETURN' in query:
                    self.strengths.append(f"[OK] Query {i+1} uses proper aggregation")
                else:
                    query_issues.append(f"Query {i+1}: Aggregation without WITH/RETURN")
            
            # 6. Check for Cartesian products
            match_count = query.count('MATCH')
            if match_count > 2 and 'WHERE' not in query:
                query_issues.append(f"Query {i+1}: Multiple MATCHes without WHERE (possible Cartesian product)")
        
        if query_issues:
            for issue in query_issues[:5]:  # Limit to first 5 issues
                self.issues.append(f"[ERROR] {issue}")
        
        # Check for query complexity
        complex_queries = [q for q in queries if len(q) > 500]
        if complex_queries:
            self.warnings.append(f"[WARN] {len(complex_queries)} complex queries (>500 chars) may need optimization")
            
    def _check_error_handling(self, source: str):
        """Check error handling patterns."""
        logger.info("  Checking error handling...")
        
        # Count try-except blocks
        try_count = source.count('try:')
        except_count = source.count('except')
        
        self.metrics['error_handling'] = {
            'try_blocks': try_count,
            'except_blocks': except_count
        }
        
        if try_count > 10:
            self.strengths.append(f"[OK] Comprehensive error handling ({try_count} try blocks)")
        elif try_count > 5:
            self.strengths.append(f"[OK] Good error handling ({try_count} try blocks)")
        else:
            self.warnings.append(f"[WARN] Limited error handling ({try_count} try blocks)")
        
        # Check for proper logging
        if 'logger.error' in source:
            self.strengths.append("[OK] Error logging implemented")
        else:
            self.issues.append("[ERROR] No error logging found")
            
        # Check for connection cleanup
        if 'driver.close()' in source or 'session.close()' in source:
            self.strengths.append("[OK] Proper connection cleanup")
        else:
            self.warnings.append("[WARN] May lack proper connection cleanup")
            
    def _validate_pipeline_integration(self, source: str):
        """Validate integration with GNN pipeline."""
        logger.info("  Validating pipeline integration...")
        
        # Check for proper data extraction methods
        extraction_methods = [
            'get_buildings_by_cable_group',
            'get_grid_topology',
            'get_building_time_series',
            'get_energy_states'
        ]
        
        for method in extraction_methods:
            if method in source:
                self.strengths.append(f"[OK] {method} implemented for GNN pipeline")
            else:
                self.warnings.append(f"[WARN] {method} not found")
        
        # Check for numpy/pandas integration
        if 'import numpy' in source or 'import pandas' in source:
            self.strengths.append("[OK] Data science library integration")
        else:
            self.issues.append("[ERROR] No numpy/pandas integration found")
            
        # Check return types
        if 'Dict[str, np.ndarray]' in source or '-> np.ndarray' in source:
            self.strengths.append("[OK] Returns numpy arrays for GNN")
        
        if '-> pd.DataFrame' in source or 'pd.DataFrame(' in source:
            self.strengths.append("[OK] Returns DataFrames for analysis")
            
    def _check_performance_patterns(self, source: str):
        """Check for performance optimization patterns."""
        logger.info("  Checking performance patterns...")
        
        # Check for batch operations
        if 'collect(' in source:
            collect_count = source.count('collect(')
            self.strengths.append(f"[OK] Uses batch collection ({collect_count} instances)")
        
        # Check for proper indexing mentions
        if 'index' in source.lower() or 'INDEX' in source:
            self.strengths.append("[OK] Index awareness in implementation")
        
        # Check for query limits
        if 'LIMIT' in source:
            self.strengths.append("[OK] Uses LIMIT for query optimization")
        
        # Check for proper WITH clause usage
        with_count = source.count('WITH')
        if with_count > 10:
            self.strengths.append(f"[OK] Extensive use of WITH for query composition ({with_count} instances)")
        
        # Check for potential N+1 query issues
        if 'for' in source and 'session.run' in source:
            # Simple heuristic: check if queries are run in loops
            lines = source.split('\n')
            for i, line in enumerate(lines):
                if 'for' in line:
                    # Check next 10 lines for session.run
                    for j in range(i+1, min(i+10, len(lines))):
                        if 'session.run' in lines[j]:
                            self.warnings.append("[WARN] Potential N+1 query pattern detected")
                            break
                            
    def analyze_query_performance(self):
        """Analyze specific query performance characteristics."""
        logger.info("Analyzing query performance patterns...")
        
        # Read the connector again for detailed analysis
        with open('data/kg_connector.py', 'r') as f:
            source = f.read()
        
        # Extract specific queries for analysis
        critical_queries = {
            'get_grid_topology': self._analyze_topology_query,
            'get_building_time_series': self._analyze_timeseries_query,
            'aggregate_to_cable_group': self._analyze_aggregation_query
        }
        
        for method_name, analyzer in critical_queries.items():
            # Find method in source
            method_pattern = f'def {method_name}.*?(?=def |$)'
            method_match = re.search(method_pattern, source, re.DOTALL)
            if method_match:
                analyzer(method_match.group())
                
    def _analyze_topology_query(self, method_source: str):
        """Analyze the grid topology query."""
        logger.info("  Analyzing grid topology query...")
        
        # Check for proper node collection
        if 'collect(DISTINCT' in method_source:
            self.strengths.append("[OK] Topology query uses DISTINCT for deduplication")
        
        # Check for edge relationship patterns
        edge_patterns = ['building_to_cable', 'cable_to_transformer', 'transformer_to_substation']
        found_patterns = sum(1 for p in edge_patterns if p in method_source)
        if found_patterns == len(edge_patterns):
            self.strengths.append("[OK] Complete edge relationship mapping")
        else:
            self.warnings.append(f"[WARN] Only {found_patterns}/{len(edge_patterns)} edge types mapped")
            
    def _analyze_timeseries_query(self, method_source: str):
        """Analyze time series query performance."""
        logger.info("  Analyzing time series query...")
        
        # Check for time range filtering
        if 'timestamp >' in method_source and 'timestamp <' in method_source:
            self.strengths.append("[OK] Time series query uses timestamp filtering")
        else:
            self.issues.append("[ERROR] Time series query lacks proper timestamp filtering")
        
        # Check for ORDER BY
        if 'ORDER BY' in method_source:
            self.strengths.append("[OK] Time series properly ordered")
        else:
            self.warnings.append("[WARN] Time series may not be properly ordered")
            
        # Check for COALESCE usage
        if 'COALESCE' in method_source:
            coalesce_count = method_source.count('COALESCE')
            self.strengths.append(f"[OK] Handles null values with COALESCE ({coalesce_count} instances)")
            
    def _analyze_aggregation_query(self, method_source: str):
        """Analyze aggregation query patterns."""
        logger.info("  Analyzing aggregation queries...")
        
        # Check for proper aggregation functions
        agg_functions = ['sum(', 'avg(', 'count(', 'reduce(']
        used_functions = [f for f in agg_functions if f in method_source]
        
        if len(used_functions) >= 2:
            self.strengths.append(f"[OK] Uses multiple aggregation functions: {', '.join(used_functions)}")
        elif used_functions:
            self.strengths.append(f"[OK] Uses aggregation: {used_functions[0]}")
        else:
            self.warnings.append("[WARN] Limited aggregation in queries")
            
    def generate_report(self):
        """Generate comprehensive validation report."""
        print("\n" + "="*80)
        print("KNOWLEDGE GRAPH IMPLEMENTATION VALIDATION REPORT")
        print("="*80)
        
        print("\n## METRICS")
        print("-" * 40)
        for key, value in self.metrics.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for k, v in value.items():
                    print(f"  - {k}: {v}")
            else:
                print(f"  - {key}: {value}")
        
        print("\n## STRENGTHS (" + str(len(self.strengths)) + " found)")
        print("-" * 40)
        for strength in self.strengths[:20]:  # Limit output
            print(strength)
        if len(self.strengths) > 20:
            print(f"  ... and {len(self.strengths) - 20} more")
        
        print("\n## WARNINGS (" + str(len(self.warnings)) + " found)")
        print("-" * 40)
        for warning in self.warnings[:10]:
            print(warning)
        if len(self.warnings) > 10:
            print(f"  ... and {len(self.warnings) - 10} more")
        
        print("\n## CRITICAL ISSUES (" + str(len(self.issues)) + " found)")
        print("-" * 40)
        if self.issues:
            for issue in self.issues:
                print(issue)
        else:
            print("[OK] No critical issues found")
        
        print("\n## RECOMMENDATIONS")
        print("-" * 40)
        
        recommendations = []
        
        # Generate recommendations based on findings
        if len(self.issues) > 0:
            recommendations.append("1. Address critical issues before production deployment")
        
        if 'total_queries' in self.metrics and self.metrics['total_queries'] > 20:
            recommendations.append("2. Consider query caching for frequently used patterns")
        
        if any('N+1' in w for w in self.warnings):
            recommendations.append("3. Refactor queries to avoid N+1 patterns using batch operations")
        
        if any('index' not in s.lower() for s in self.strengths):
            recommendations.append("4. Implement proper indexing strategy for key properties")
        
        if not any('cache' in s.lower() for s in self.strengths):
            recommendations.append("5. Consider implementing query result caching")
        
        recommendations.append("6. Add query execution time monitoring")
        recommendations.append("7. Implement connection pooling for high-load scenarios")
        recommendations.append("8. Add comprehensive integration tests for all query methods")
        
        for rec in recommendations:
            print(f"  {rec}")
        
        print("\n## PERFORMANCE ANALYSIS")
        print("-" * 40)
        print("Query Complexity Assessment:")
        if self.metrics.get('total_queries', 0) > 0:
            print(f"  - Total queries analyzed: {self.metrics['total_queries']}")
            print(f"  - Error handling coverage: {'Good' if self.metrics.get('error_handling', {}).get('try_blocks', 0) > 10 else 'Needs improvement'}")
            print(f"  - Pipeline integration: {'Complete' if len([s for s in self.strengths if 'pipeline' in s.lower()]) > 2 else 'Partial'}")
        
        print("\n## SCHEMA VALIDATION")
        print("-" * 40)
        print("Energy System Schema Coverage:")
        node_coverage = len([s for s in self.strengths if 'node type implemented' in s])
        rel_coverage = len([s for s in self.strengths if 'relationship implemented' in s])
        print(f"  - Node types: {node_coverage}/10 implemented")
        print(f"  - Relationships: {rel_coverage}/7 implemented")
        print(f"  - Hierarchy: {'Valid' if any('hierarchy' in s for s in self.strengths) else 'Needs validation'}")
        
        print("\n" + "="*80)
        print("OVERALL ASSESSMENT: ", end="")
        
        # Calculate score
        score = len(self.strengths) - len(self.warnings) - (len(self.issues) * 3)
        if score > 15 and len(self.issues) == 0:
            print("EXCELLENT - Production Ready")
        elif score > 10 and len(self.issues) < 3:
            print("GOOD - Minor improvements needed")
        elif score > 5:
            print("FAIR - Significant improvements recommended")
        else:
            print("NEEDS WORK - Major refactoring required")
        
        print("="*80)
        
        return {
            'strengths': self.strengths,
            'warnings': self.warnings,
            'issues': self.issues,
            'metrics': self.metrics,
            'score': score
        }

def main():
    """Run the comprehensive KG validation."""
    validator = KGImplementationValidator()
    
    try:
        # Analyze implementation
        validator.analyze_kg_connector()
        validator.analyze_query_performance()
        
        # Generate report
        results = validator.generate_report()
        
        # Write detailed report
        with open('KG_VALIDATION_REPORT.txt', 'w') as f:
            f.write("KNOWLEDGE GRAPH IMPLEMENTATION VALIDATION\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("DETAILED FINDINGS\n")
            f.write("-" * 40 + "\n")
            
            f.write("\nStrengths:\n")
            for s in results['strengths']:
                f.write(f"  {s}\n")
            
            f.write("\nWarnings:\n")
            for w in results['warnings']:
                f.write(f"  {w}\n")
            
            f.write("\nIssues:\n")
            for i in results['issues']:
                f.write(f"  {i}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"Final Score: {results['score']}\n")
        
        print("\n[OK] Detailed report saved to KG_VALIDATION_REPORT.txt")
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)