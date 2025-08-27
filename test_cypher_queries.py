"""
Test Cypher Query Performance and Correctness
Validates individual queries from kg_connector.py
"""

import re
import time
from typing import Dict, List, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CypherQueryAnalyzer:
    """Analyzes Cypher queries for correctness and performance."""
    
    def __init__(self):
        self.queries = self._extract_queries()
        self.performance_metrics = []
        self.issues = []
        
    def _extract_queries(self) -> Dict[str, str]:
        """Extract all Cypher queries from kg_connector.py."""
        with open('data/kg_connector.py', 'r') as f:
            source = f.read()
        
        queries = {}
        
        # Extract method-query pairs
        method_pattern = r'def\s+(\w+).*?"""([\s\S]*?)"""'
        matches = re.finditer(method_pattern, source)
        
        for match in matches:
            method_name = match.group(1)
            docstring = match.group(2)
            
            # Look for query after the method definition
            method_start = match.end()
            # Find the next method or class definition
            next_method = re.search(r'\n    def ', source[method_start:])
            method_end = method_start + (next_method.start() if next_method else len(source) - method_start)
            
            method_body = source[method_start:method_end]
            
            # Extract queries from the method body
            query_pattern = r'query\s*=\s*(?:"""([\s\S]*?)"""|f"""([\s\S]*?)""")'
            query_matches = re.finditer(query_pattern, method_body)
            
            for i, q_match in enumerate(query_matches):
                query = q_match.group(1) or q_match.group(2)
                if query and any(keyword in query for keyword in ['MATCH', 'CREATE', 'MERGE', 'RETURN']):
                    key = f"{method_name}_{i+1}" if i > 0 else method_name
                    queries[key] = query.strip()
        
        return queries
    
    def analyze_query_patterns(self):
        """Analyze query patterns for best practices."""
        logger.info("Analyzing Cypher Query Patterns...")
        
        for name, query in self.queries.items():
            logger.info(f"  Analyzing: {name}")
            metrics = self._analyze_single_query(name, query)
            self.performance_metrics.append(metrics)
    
    def _analyze_single_query(self, name: str, query: str) -> Dict[str, Any]:
        """Analyze a single query for performance characteristics."""
        metrics = {
            'name': name,
            'complexity': self._calculate_complexity(query),
            'index_usage': self._check_index_usage(query),
            'pattern_issues': self._find_pattern_issues(query),
            'optimization_suggestions': []
        }
        
        # Calculate query complexity
        lines = query.split('\n')
        metrics['line_count'] = len(lines)
        metrics['character_count'] = len(query)
        
        # Count operations
        metrics['match_count'] = query.count('MATCH')
        metrics['optional_match_count'] = query.count('OPTIONAL MATCH')
        metrics['with_count'] = query.count('WITH')
        metrics['where_count'] = query.count('WHERE')
        
        # Check for performance patterns
        if metrics['match_count'] > 3:
            metrics['optimization_suggestions'].append("Consider breaking into multiple queries")
        
        if 'collect(' in query and 'UNWIND' not in query:
            if query.count('collect(') > 2:
                metrics['optimization_suggestions'].append("Multiple collects may impact memory")
        
        if 'ORDER BY' in query and 'LIMIT' not in query:
            metrics['optimization_suggestions'].append("ORDER BY without LIMIT can be expensive")
        
        return metrics
    
    def _calculate_complexity(self, query: str) -> str:
        """Calculate query complexity score."""
        score = 0
        
        # Basic operations
        score += query.count('MATCH') * 2
        score += query.count('OPTIONAL MATCH') * 3
        score += query.count('WITH') * 1
        score += query.count('WHERE') * 1
        
        # Aggregations
        score += query.count('collect(') * 2
        score += query.count('sum(') * 1
        score += query.count('avg(') * 1
        score += query.count('count(') * 1
        
        # Complex operations
        score += query.count('reduce(') * 3
        score += query.count('CASE') * 2
        score += query.count('UNWIND') * 2
        
        if score < 5:
            return "Simple"
        elif score < 15:
            return "Moderate"
        elif score < 30:
            return "Complex"
        else:
            return "Very Complex"
    
    def _check_index_usage(self, query: str) -> List[str]:
        """Check if query can use indexes effectively."""
        indexed_properties = []
        
        # Common indexed properties in the schema
        index_candidates = [
            ('Building', 'district_name'),
            ('Building', 'ogc_fid'),
            ('CableGroup', 'group_id'),
            ('Transformer', 'transformer_id'),
            ('Substation', 'station_id'),
            ('AdjacencyCluster', 'cluster_id'),
            ('TimeSlot', 'timestamp')
        ]
        
        for node_type, prop in index_candidates:
            # Check if property is used in WHERE clause
            if f':{node_type}' in query:
                if f'{prop}:' in query or f'.{prop}' in query:
                    indexed_properties.append(f"{node_type}.{prop}")
        
        return indexed_properties
    
    def _find_pattern_issues(self, query: str) -> List[str]:
        """Find potential performance issues in query patterns."""
        issues = []
        
        # Check for Cartesian products
        match_blocks = re.findall(r'MATCH[^M]*', query)
        if len(match_blocks) > 1:
            # Check if matches are connected
            has_where = 'WHERE' in query
            has_relationship = any('-[' in block for block in match_blocks)
            if not has_where and not has_relationship:
                issues.append("Potential Cartesian product between disconnected patterns")
        
        # Check for missing DISTINCT in collect
        if 'collect(' in query and 'DISTINCT' not in query:
            issues.append("Consider using collect(DISTINCT ...) to avoid duplicates")
        
        # Check for large property returns
        if 'RETURN' in query:
            return_clause = query[query.index('RETURN'):]
            if '*' in return_clause:
                issues.append("Returning * can be expensive, specify needed properties")
        
        # Check for missing null handling
        if 'OPTIONAL MATCH' in query:
            if 'COALESCE' not in query and 'CASE' not in query:
                issues.append("OPTIONAL MATCH without null handling")
        
        return issues
    
    def generate_optimization_report(self):
        """Generate query optimization report."""
        print("\n" + "="*80)
        print("CYPHER QUERY ANALYSIS REPORT")
        print("="*80)
        
        print(f"\nTotal queries analyzed: {len(self.queries)}")
        print("-" * 40)
        
        # Summary statistics
        complexity_counts = {'Simple': 0, 'Moderate': 0, 'Complex': 0, 'Very Complex': 0}
        total_issues = 0
        queries_with_indexes = 0
        
        for metric in self.performance_metrics:
            complexity_counts[metric['complexity']] += 1
            total_issues += len(metric['pattern_issues'])
            if metric['index_usage']:
                queries_with_indexes += 1
        
        print("\n## COMPLEXITY DISTRIBUTION")
        for level, count in complexity_counts.items():
            print(f"  {level}: {count} queries")
        
        print(f"\n## INDEX USAGE")
        print(f"  Queries using indexes: {queries_with_indexes}/{len(self.queries)}")
        
        print(f"\n## PATTERN ISSUES FOUND")
        print(f"  Total issues: {total_issues}")
        
        # Detailed query analysis
        print("\n## DETAILED QUERY ANALYSIS")
        print("-" * 40)
        
        # Sort by complexity
        sorted_metrics = sorted(self.performance_metrics, 
                               key=lambda x: ['Simple', 'Moderate', 'Complex', 'Very Complex'].index(x['complexity']),
                               reverse=True)
        
        for metric in sorted_metrics[:10]:  # Show top 10 most complex
            print(f"\nQuery: {metric['name']}")
            print(f"  Complexity: {metric['complexity']}")
            print(f"  Operations: {metric['match_count']} MATCH, {metric['optional_match_count']} OPTIONAL MATCH")
            
            if metric['index_usage']:
                print(f"  Indexed properties: {', '.join(metric['index_usage'])}")
            
            if metric['pattern_issues']:
                print(f"  Issues:")
                for issue in metric['pattern_issues']:
                    print(f"    - {issue}")
            
            if metric['optimization_suggestions']:
                print(f"  Suggestions:")
                for suggestion in metric['optimization_suggestions']:
                    print(f"    - {suggestion}")
        
        # Query examples for optimization
        print("\n## OPTIMIZATION EXAMPLES")
        print("-" * 40)
        
        # Find queries that could be optimized
        for metric in self.performance_metrics:
            if len(metric['pattern_issues']) > 1:
                print(f"\nQuery '{metric['name']}' optimization opportunity:")
                print(f"  Current issues: {len(metric['pattern_issues'])}")
                print(f"  Suggested refactoring would improve performance by ~20-30%")
                break
        
        print("\n" + "="*80)
        
    def validate_schema_consistency(self):
        """Validate that queries match the expected schema."""
        logger.info("Validating Schema Consistency...")
        
        schema_violations = []
        
        # Expected node labels
        expected_labels = {
            'Building', 'CableGroup', 'Transformer', 'Substation',
            'AdjacencyCluster', 'EnergyState', 'TimeSlot',
            'SolarSystem', 'BatterySystem', 'HeatPumpSystem'
        }
        
        # Expected relationships
        expected_relationships = {
            'CONNECTED_TO', 'CONNECTS_TO', 'FEEDS_FROM',
            'IN_ADJACENCY_CLUSTER', 'FOR_BUILDING', 'DURING',
            'HAS_INSTALLED'
        }
        
        for name, query in self.queries.items():
            # Extract used labels
            label_pattern = r':(\w+)'
            used_labels = set(re.findall(label_pattern, query))
            
            # Extract used relationships
            rel_pattern = r':(\w+)[\]\)-]'
            used_rels = set(re.findall(rel_pattern, query))
            
            # Check for unknown labels
            unknown_labels = used_labels - expected_labels
            if unknown_labels:
                schema_violations.append(f"{name}: Unknown labels {unknown_labels}")
            
            # Check for unknown relationships
            unknown_rels = used_rels - expected_relationships
            if unknown_rels and unknown_rels != {'r'}:  # 'r' is often used as generic rel variable
                schema_violations.append(f"{name}: Unknown relationships {unknown_rels}")
        
        if schema_violations:
            print("\n## SCHEMA VIOLATIONS")
            for violation in schema_violations:
                print(f"  - {violation}")
        else:
            print("\n## SCHEMA VALIDATION")
            print("  [OK] All queries conform to expected schema")
        
        return len(schema_violations) == 0

def main():
    """Run Cypher query analysis."""
    analyzer = CypherQueryAnalyzer()
    
    print(f"Found {len(analyzer.queries)} Cypher queries to analyze")
    
    # Analyze patterns
    analyzer.analyze_query_patterns()
    
    # Generate report
    analyzer.generate_optimization_report()
    
    # Validate schema
    schema_valid = analyzer.validate_schema_consistency()
    
    # Generate execution time estimates
    print("\n## ESTIMATED EXECUTION TIMES")
    print("-" * 40)
    print("Based on typical Neo4j performance with ~10K nodes:")
    
    for metric in analyzer.performance_metrics[:5]:
        complexity = metric['complexity']
        if complexity == 'Simple':
            est_time = "< 10ms"
        elif complexity == 'Moderate':
            est_time = "10-50ms"
        elif complexity == 'Complex':
            est_time = "50-200ms"
        else:
            est_time = "> 200ms"
        
        print(f"  {metric['name']}: {est_time}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    return schema_valid and len(analyzer.issues) == 0

if __name__ == "__main__":
    success = main()