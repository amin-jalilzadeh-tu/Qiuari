# inference/query_processor.py
"""
Natural language query processor for energy GNN system
Maps user queries to appropriate tasks and extracts parameters
"""

import re
from typing import Dict, List, Tuple, Optional, Any
import spacy
from dataclasses import dataclass
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class TaskType(Enum):
    """Enumeration of available tasks"""
    CLUSTERING = "clustering"
    SOLAR = "solar_optimization"
    RETROFIT = "retrofit"
    THERMAL = "thermal_sharing"
    ELECTRIFICATION = "electrification"
    BATTERY = "battery_placement"
    P2P_TRADING = "p2p_trading"
    CONGESTION = "congestion_prediction"
    MULTI_TASK = "multi_task"
    UNKNOWN = "unknown"

@dataclass
class QueryIntent:
    """Structured query intent"""
    task: TaskType
    parameters: Dict[str, Any]
    constraints: Dict[str, Any]
    confidence: float
    original_query: str

class QueryProcessor:
    """Process natural language queries and map to tasks"""
    
    def __init__(self, config: Dict):
        """
        Initialize query processor
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Load NLP model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            logger.warning("Spacy model not found, using rule-based processing")
            self.nlp = None
        
        # Define keyword mappings for tasks
        self.task_keywords = {
            TaskType.CLUSTERING: [
                'cluster', 'community', 'group', 'aggregate', 'combine',
                'energy community', 'microgrid', 'collective'
            ],
            TaskType.SOLAR: [
                'solar', 'pv', 'photovoltaic', 'panel', 'renewable',
                'generation', 'rooftop', 'sun'
            ],
            TaskType.RETROFIT: [
                'retrofit', 'renovation', 'insulation', 'efficiency',
                'upgrade', 'improvement', 'refurbishment', 'energy waste'
            ],
            TaskType.THERMAL: [
                'heat', 'thermal', 'heating', 'cooling', 'temperature',
                'hvac', 'district heating', 'heat sharing'
            ],
            TaskType.ELECTRIFICATION: [
                'heat pump', 'electrification', 'electrify', 'electric heating',
                'hp', 'ashp', 'gshp', 'electric'
            ],
            TaskType.BATTERY: [
                'battery', 'storage', 'energy storage', 'ess', 'bess',
                'backup', 'resilience'
            ],
            TaskType.P2P_TRADING: [
                'trading', 'p2p', 'peer to peer', 'energy trading',
                'share', 'exchange', 'market', 'prosumer'
            ],
            TaskType.CONGESTION: [
                'congestion', 'overload', 'capacity', 'grid stress',
                'bottleneck', 'constraint', 'limitation', 'peak'
            ]
        }
        
        # Define parameter extraction patterns
        self.parameter_patterns = {
            'capacity': r'(\d+\.?\d*)\s*(kw|mw|kwp|mwp|kwh|mwh)',
            'percentage': r'(\d+\.?\d*)\s*%',
            'money': r'[€$£]\s*(\d+\.?\d*[km]?)',
            'time': r'(\d+)\s*(year|month|day|hour)s?',
            'area': r'(\d+\.?\d*)\s*(m2|m²|sqm|square meter)',
            'building_id': r'building\s*#?(\d+)',
            'location': r'(lv|mv|hv|transformer|substation)\s*(\w+)'
        }
        
        # Question type patterns
        self.question_patterns = {
            'what': ['what', 'which'],
            'where': ['where', 'location'],
            'when': ['when', 'time', 'schedule'],
            'how': ['how', 'method', 'approach'],
            'why': ['why', 'reason', 'cause'],
            'optimization': ['best', 'optimal', 'maximize', 'minimize', 'optimize']
        }
        
        logger.info("Initialized QueryProcessor")
    
    def process(self, query: str) -> QueryIntent:
        """
        Process natural language query
        
        Args:
            query: User's natural language query
            
        Returns:
            Structured query intent
        """
        query_lower = query.lower()
        
        # Identify task type
        task, confidence = self._identify_task(query_lower)
        
        # Extract parameters
        parameters = self._extract_parameters(query_lower)
        
        # Extract constraints
        constraints = self._extract_constraints(query_lower)
        
        # Handle compound queries
        if self._is_compound_query(query_lower):
            task = TaskType.MULTI_TASK
            parameters['sub_tasks'] = self._extract_sub_tasks(query_lower)
        
        # Create query intent
        intent = QueryIntent(
            task=task,
            parameters=parameters,
            constraints=constraints,
            confidence=confidence,
            original_query=query
        )
        
        logger.info(f"Processed query: {task.value} with confidence {confidence:.2f}")
        
        return intent
    
    def _identify_task(self, query: str) -> Tuple[TaskType, float]:
        """Identify the primary task from query"""
        task_scores = {}
        
        # Score each task based on keyword matches
        for task, keywords in self.task_keywords.items():
            score = 0
            matches = 0
            
            for keyword in keywords:
                if keyword in query:
                    # Longer keywords get higher scores
                    score += len(keyword.split())
                    matches += 1
            
            if matches > 0:
                # Normalize by number of keywords
                task_scores[task] = score / len(keywords)
        
        if not task_scores:
            return TaskType.UNKNOWN, 0.0
        
        # Get best matching task
        best_task = max(task_scores, key=task_scores.get)
        confidence = min(task_scores[best_task] / 2, 1.0)  # Scale confidence
        
        # Check for specific patterns that boost confidence
        if self._has_specific_pattern(query, best_task):
            confidence = min(confidence + 0.2, 1.0)
        
        return best_task, confidence
    
    def _extract_parameters(self, query: str) -> Dict[str, Any]:
        """Extract parameters from query"""
        parameters = {}
        
        # Extract numerical parameters
        for param_name, pattern in self.parameter_patterns.items():
            matches = re.findall(pattern, query, re.IGNORECASE)
            if matches:
                if param_name == 'capacity':
                    value, unit = matches[0]
                    # Convert to standard unit (kW)
                    if 'mw' in unit.lower():
                        value = float(value) * 1000
                    parameters[param_name] = float(value)
                    parameters[f'{param_name}_unit'] = unit
                
                elif param_name == 'money':
                    value = matches[0]
                    # Handle k/m suffixes
                    if 'k' in value.lower():
                        value = float(value.replace('k', '')) * 1000
                    elif 'm' in value.lower():
                        value = float(value.replace('m', '')) * 1000000
                    parameters['budget'] = float(value)
                
                elif param_name == 'time':
                    value, unit = matches[0]
                    parameters['time_horizon'] = int(value)
                    parameters['time_unit'] = unit
                
                else:
                    parameters[param_name] = matches[0]
        
        # Extract specific entities
        if 'all buildings' in query:
            parameters['scope'] = 'all'
        elif 'building' in query:
            building_matches = re.findall(r'building[s]?\s+(\d+(?:\s*,\s*\d+)*)', query)
            if building_matches:
                building_ids = [int(b.strip()) for b in building_matches[0].split(',')]
                parameters['building_ids'] = building_ids
        
        # Extract optimization objectives
        if 'maximize' in query:
            if 'self-sufficiency' in query or 'self sufficiency' in query:
                parameters['objective'] = 'maximize_self_sufficiency'
            elif 'savings' in query:
                parameters['objective'] = 'maximize_savings'
        elif 'minimize' in query:
            if 'cost' in query:
                parameters['objective'] = 'minimize_cost'
            elif 'peak' in query:
                parameters['objective'] = 'minimize_peak'
        
        return parameters
    
    def _extract_constraints(self, query: str) -> Dict[str, Any]:
        """Extract constraints from query"""
        constraints = {}
        
        # Budget constraints
        if 'budget' in query or 'less than' in query or 'under' in query:
            money_match = re.search(r'[€$£]\s*(\d+\.?\d*[km]?)', query)
            if money_match:
                value = money_match.group(1)
                if 'k' in value.lower():
                    value = float(value.replace('k', '')) * 1000
                elif 'm' in value.lower():
                    value = float(value.replace('m', '')) * 1000000
                constraints['max_budget'] = float(value)
        
        # Time constraints
        if 'within' in query:
            time_match = re.search(r'within\s+(\d+)\s*(year|month|day)s?', query)
            if time_match:
                constraints['max_time'] = int(time_match.group(1))
                constraints['time_unit'] = time_match.group(2)
        
        # ROI constraints
        if 'payback' in query or 'roi' in query:
            roi_match = re.search(r'(\d+)\s*year', query)
            if roi_match:
                constraints['max_payback_years'] = int(roi_match.group(1))
        
        # Grid constraints
        if 'grid' in query:
            if 'no upgrade' in query or 'without upgrade' in query:
                constraints['grid_upgrade'] = False
            if 'respect' in query or 'within' in query:
                constraints['respect_grid_limits'] = True
        
        # Transformer constraints
        if 'same transformer' in query:
            constraints['same_transformer'] = True
        
        return constraints
    
    def _is_compound_query(self, query: str) -> bool:
        """Check if query involves multiple tasks"""
        task_count = 0
        
        for keywords in self.task_keywords.values():
            if any(keyword in query for keyword in keywords):
                task_count += 1
        
        return task_count > 1
    
    def _extract_sub_tasks(self, query: str) -> List[TaskType]:
        """Extract multiple tasks from compound query"""
        sub_tasks = []
        
        for task, keywords in self.task_keywords.items():
            if any(keyword in query for keyword in keywords):
                sub_tasks.append(task)
        
        return sub_tasks
    
    def _has_specific_pattern(self, query: str, task: TaskType) -> bool:
        """Check for task-specific patterns"""
        patterns = {
            TaskType.CLUSTERING: [
                'form.*communit',
                'create.*cluster',
                'group.*building'
            ],
            TaskType.SOLAR: [
                'install.*solar',
                'solar.*potential',
                'where.*panel'
            ],
            TaskType.RETROFIT: [
                'need.*retrofit',
                'energy.*waste',
                'improve.*efficiency'
            ]
        }
        
        if task in patterns:
            for pattern in patterns[task]:
                if re.search(pattern, query):
                    return True
        
        return False
    
    def parse_complex_query(self, query: str) -> List[QueryIntent]:
        """Parse complex multi-step queries"""
        intents = []
        
        # Split by conjunctions
        sub_queries = re.split(r'\s+(?:and|then|also|plus)\s+', query, flags=re.IGNORECASE)
        
        for sub_query in sub_queries:
            intent = self.process(sub_query)
            if intent.task != TaskType.UNKNOWN:
                intents.append(intent)
        
        return intents
    
    def suggest_query(self, partial_query: str) -> List[str]:
        """Suggest query completions"""
        suggestions = []
        
        # Common query templates
        templates = [
            "What buildings should get solar panels?",
            "Form energy communities with maximum self-sufficiency",
            "Which buildings need retrofitting?",
            "Where should we install batteries?",
            "Find best locations for heat pumps",
            "Predict grid congestion for next year",
            "Optimize P2P energy trading pairs",
            "What is the ROI for solar on building {id}?",
            "Create clusters respecting transformer boundaries",
            "How can we achieve 80% renewable energy?"
        ]
        
        # Filter templates based on partial query
        partial_lower = partial_query.lower()
        for template in templates:
            if partial_lower in template.lower():
                suggestions.append(template)
        
        # Add contextual suggestions based on identified task
        task, _ = self._identify_task(partial_lower)
        if task != TaskType.UNKNOWN:
            task_suggestions = self._get_task_suggestions(task)
            suggestions.extend(task_suggestions)
        
        return suggestions[:5]  # Return top 5 suggestions
    
    def _get_task_suggestions(self, task: TaskType) -> List[str]:
        """Get task-specific query suggestions"""
        suggestions = {
            TaskType.CLUSTERING: [
                "Form 5 energy communities with maximum peak reduction",
                "Create clusters of 10-15 buildings each",
                "Group buildings by complementary consumption patterns"
            ],
            TaskType.SOLAR: [
                "Find top 10 buildings for solar installation",
                "Calculate solar potential for all south-facing roofs",
                "What's the total solar capacity we can install?"
            ],
            TaskType.RETROFIT: [
                "Identify buildings with energy label below C",
                "Which retrofits give best ROI?",
                "Prioritize retrofits with €100k budget"
            ]
        }
        
        return suggestions.get(task, [])

class QueryValidator:
    """Validate and sanitize user queries"""
    
    def __init__(self):
        """Initialize query validator"""
        self.max_query_length = 500
        self.forbidden_patterns = [
            r'drop\s+table',
            r'delete\s+from',
            r'update\s+set',
            r';\s*--'
        ]
    
    def validate(self, query: str) -> Tuple[bool, str]:
        """
        Validate query for safety and feasibility
        
        Args:
            query: User query
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check length
        if len(query) > self.max_query_length:
            return False, f"Query too long (max {self.max_query_length} characters)"
        
        # Check for SQL injection patterns
        for pattern in self.forbidden_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return False, "Query contains forbidden patterns"
        
        # Check for minimum content
        if len(query.strip()) < 3:
            return False, "Query too short"
        
        return True, ""

class QueryResponseBuilder:
    """Build natural language responses from results"""
    
    def __init__(self):
        """Initialize response builder"""
        self.response_templates = {
            TaskType.CLUSTERING: {
                'success': "I've identified {num_clusters} optimal energy communities. "
                          "The configuration achieves {peak_reduction:.1%} peak reduction "
                          "and {self_sufficiency:.1%} self-sufficiency.",
                'partial': "I've formed {num_clusters} communities, but some buildings "
                          "couldn't be assigned due to {reason}.",
                'failure': "Unable to form communities: {reason}"
            },
            TaskType.SOLAR: {
                'success': "Recommended {num_installations} solar installations totaling "
                          "{capacity:.1f} kWp. Expected ROI: {roi:.1f} years. "
                          "Annual generation: {generation:.0f} MWh.",
                'partial': "Found {num_installations} viable solar locations, "
                          "but {constraints} limit full potential.",
                'failure': "No suitable locations found for solar: {reason}"
            }
        }
    
    def build_response(self, task: TaskType, results: Dict, 
                       status: str = 'success') -> str:
        """
        Build natural language response
        
        Args:
            task: Task type
            results: Task results
            status: Result status (success/partial/failure)
            
        Returns:
            Natural language response
        """
        if task not in self.response_templates:
            return f"Completed {task.value} task. See detailed results below."
        
        template = self.response_templates[task].get(
            status, 
            "Task completed with status: {status}"
        )
        
        try:
            response = template.format(**results)
        except KeyError:
            response = f"Task {task.value} completed. "
            response += f"Key metrics: {', '.join(f'{k}={v}' for k, v in results.items() if not isinstance(v, (dict, list)))}"
        
        return response

# Usage example
if __name__ == "__main__":
    import yaml
    
    # Load configuration
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create query processor
    processor = QueryProcessor(config)
    
    # Test queries
    test_queries = [
        "What buildings should get solar panels?",
        "Form energy communities with at least 80% self-sufficiency",
        "Which buildings need retrofitting within €100k budget?",
        "Find optimal battery locations for peak shaving",
        "Where should we install heat pumps first?",
        "Predict grid congestion for next year",
        "Create P2P trading pairs between buildings 1, 5, and 10",
        "Optimize everything with €500k budget"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        intent = processor.process(query)
        print(f"  Task: {intent.task.value}")
        print(f"  Parameters: {intent.parameters}")
        print(f"  Constraints: {intent.constraints}")
        print(f"  Confidence: {intent.confidence:.2f}")