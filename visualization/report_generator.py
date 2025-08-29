"""
Report Generation System
Creates comprehensive reports in multiple formats
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import json
from typing import Dict, List, Optional, Any
from jinja2 import Template
import logging
from dataclasses import asdict

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates various types of reports"""
    
    def __init__(self, output_dir: str = "results/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Report templates
        self.templates = {
            'executive': self._get_executive_template(),
            'technical': self._get_technical_template(),
            'stakeholder': self._get_stakeholder_template()
        }
    
    def generate_executive_summary(self, metrics: Any, 
                                  comparison: Optional[Dict] = None) -> str:
        """Generate executive summary report"""
        
        template = Template(self.templates['executive'])
        
        # Prepare data
        data = {
            'date': datetime.now().strftime('%B %d, %Y'),
            'num_clusters': metrics.num_clusters,
            'num_buildings': metrics.total_clustered_buildings if hasattr(metrics, 'total_clustered_buildings') else int(metrics.num_clusters * metrics.avg_cluster_size) if metrics.num_clusters > 0 else 0,
            'self_sufficiency': f"{metrics.avg_self_sufficiency:.1%}",
            'cost_savings': f"€{metrics.total_cost_savings_eur:,.0f}",
            'carbon_reduction': f"{metrics.carbon_reduction_tons:.1f}",
            'peak_reduction': f"{metrics.total_peak_reduction:.1%}",
            'solar_coverage': f"{metrics.solar_coverage_percent:.1%}",
            'roi_years': f"{metrics.avg_solar_roi_years:.1f}"
        }
        
        # Add comparison if available
        if comparison:
            data['has_comparison'] = True
            data['improvement'] = comparison
        else:
            data['has_comparison'] = False
        
        report = template.render(**data)
        
        # Save report
        report_path = self.output_dir / f"executive_summary_{datetime.now().strftime('%Y%m%d')}.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Executive summary saved to {report_path}")
        return report
    
    def generate_technical_report(self, metrics: Any, 
                                 cluster_details: Dict,
                                 network_analysis: Dict) -> str:
        """Generate detailed technical report"""
        
        template = Template(self.templates['technical'])
        
        data = {
            'date': datetime.now().strftime('%B %d, %Y'),
            'metrics': asdict(metrics),
            'clusters': cluster_details,
            'network': network_analysis,
            'timestamp': datetime.now().isoformat()
        }
        
        report = template.render(**data)
        
        # Save report
        report_path = self.output_dir / f"technical_report_{datetime.now().strftime('%Y%m%d')}.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Technical report saved to {report_path}")
        return report
    
    def generate_intervention_recommendations(self, 
                                            solar_candidates: List,
                                            battery_candidates: List,
                                            retrofit_candidates: List) -> str:
        """Generate intervention recommendations report"""
        
        report = []
        report.append("# Energy Community Intervention Recommendations")
        report.append(f"\nGenerated: {datetime.now().strftime('%B %d, %Y')}\n")
        
        # Solar recommendations
        report.append("## 🌞 Solar Installation Priorities\n")
        report.append("Top 10 buildings for solar installation based on ROI and impact:\n")
        
        for i, candidate in enumerate(solar_candidates[:10], 1):
            report.append(f"{i}. **Building {candidate.get('id', 'Unknown')}**")
            report.append(f"   - Energy Label: {candidate.get('label', 'N/A')}")
            report.append(f"   - Roof Area: {candidate.get('roof_area', 0):.0f} m²")
            report.append(f"   - Expected Capacity: {candidate.get('capacity', 0):.1f} kWp")
            report.append(f"   - ROI: {candidate.get('roi_years', 0):.1f} years")
            report.append(f"   - Priority Score: {candidate.get('priority', 0):.2f}\n")
        
        # Battery recommendations
        report.append("## 🔋 Battery Storage Priorities\n")
        report.append("Recommended battery installations for peak shaving:\n")
        
        for i, candidate in enumerate(battery_candidates[:10], 1):
            report.append(f"{i}. **Building {candidate.get('id', 'Unknown')}**")
            report.append(f"   - Peak Demand: {candidate.get('peak_demand', 0):.1f} kW")
            report.append(f"   - Recommended Capacity: {candidate.get('battery_size', 0):.0f} kWh")
            report.append(f"   - Peak Reduction: {candidate.get('peak_reduction', 0):.1%}")
            report.append(f"   - Annual Savings: €{candidate.get('annual_savings', 0):,.0f}\n")
        
        # Retrofit recommendations
        report.append("## 🏠 Energy Retrofit Priorities\n")
        report.append("Buildings requiring energy efficiency improvements:\n")
        
        for i, candidate in enumerate(retrofit_candidates[:10], 1):
            report.append(f"{i}. **Building {candidate.get('id', 'Unknown')}**")
            report.append(f"   - Current Label: {candidate.get('current_label', 'N/A')}")
            report.append(f"   - Target Label: {candidate.get('target_label', 'N/A')}")
            report.append(f"   - Estimated Cost: €{candidate.get('cost', 0):,.0f}")
            report.append(f"   - Energy Savings: {candidate.get('savings_percent', 0):.0%}\n")
        
        report_text = '\n'.join(report)
        
        # Save report
        report_path = self.output_dir / f"interventions_{datetime.now().strftime('%Y%m%d')}.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        return report_text
    
    def generate_cluster_quality_report(self, cluster_metrics: Dict) -> str:
        """Generate cluster quality assessment report"""
        
        report = []
        report.append("# Energy Community Cluster Quality Report")
        report.append(f"\nGenerated: {datetime.now().strftime('%B %d, %Y')}\n")
        
        # Summary statistics
        report.append("## Summary Statistics\n")
        
        quality_counts = {'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0}
        total_self_sufficiency = []
        total_complementarity = []
        
        for cluster_id, metrics in cluster_metrics.items():
            quality = metrics.get_quality_label()
            quality_counts[quality] += 1
            total_self_sufficiency.append(metrics.self_sufficiency_ratio)
            total_complementarity.append(metrics.complementarity_score)
        
        report.append(f"- Total Clusters: {len(cluster_metrics)}")
        report.append(f"- Excellent: {quality_counts['excellent']}")
        report.append(f"- Good: {quality_counts['good']}")
        report.append(f"- Fair: {quality_counts['fair']}")
        report.append(f"- Poor: {quality_counts['poor']}")
        report.append(f"- Average Self-Sufficiency: {np.mean(total_self_sufficiency):.1%}")
        report.append(f"- Average Complementarity: {np.mean(total_complementarity):.2f}\n")
        
        # Detailed cluster analysis
        report.append("## Cluster Details\n")
        
        # Sort clusters by quality score
        sorted_clusters = sorted(cluster_metrics.items(), 
                               key=lambda x: x[1].get_overall_score(), 
                               reverse=True)
        
        for cluster_id, metrics in sorted_clusters:
            report.append(f"### Cluster {cluster_id}")
            report.append(f"**Quality: {metrics.get_quality_label().upper()}** "
                        f"(Score: {metrics.get_overall_score():.1f}/100)\n")
            
            report.append("**Metrics:**")
            report.append(f"- Buildings: {metrics.member_count}")
            report.append(f"- LV Group: {metrics.lv_group_id}")
            report.append(f"- Self-Sufficiency: {metrics.self_sufficiency_ratio:.1%}")
            report.append(f"- Self-Consumption: {metrics.self_consumption_ratio:.1%}")
            report.append(f"- Complementarity: {metrics.complementarity_score:.2f}")
            report.append(f"- Peak Reduction: {metrics.peak_reduction_ratio:.1%}")
            report.append(f"- Temporal Stability: {metrics.temporal_stability:.1%}\n")
            
            report.append("**Energy Balance:**")
            report.append(f"- Total Demand: {metrics.total_demand_kwh:.0f} kWh")
            report.append(f"- Total Generation: {metrics.total_generation_kwh:.0f} kWh")
            report.append(f"- Shared Energy: {metrics.total_shared_kwh:.0f} kWh")
            report.append(f"- Grid Import: {metrics.grid_import_kwh:.0f} kWh")
            report.append(f"- Grid Export: {metrics.grid_export_kwh:.0f} kWh\n")
            
            report.append("**Explanation:**")
            report.append(metrics.get_explanation())
            report.append("\n---\n")
        
        report_text = '\n'.join(report)
        
        # Save report
        report_path = self.output_dir / f"cluster_quality_{datetime.now().strftime('%Y%m%d')}.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        return report_text
    
    def generate_stakeholder_report(self, metrics: Any, 
                                   benefits: Dict,
                                   next_steps: List[str]) -> str:
        """Generate stakeholder-friendly report"""
        
        template = Template(self.templates['stakeholder'])
        
        data = {
            'date': datetime.now().strftime('%B %d, %Y'),
            'community_size': int(metrics.num_lv_groups * metrics.avg_buildings_per_lv),
            'annual_savings': f"€{metrics.total_cost_savings_eur * 12:,.0f}",
            'carbon_saved': f"{metrics.carbon_reduction_tons:.0f}",
            'green_energy': f"{metrics.total_generation_mwh:.0f}",
            'benefits': benefits,
            'next_steps': next_steps
        }
        
        report = template.render(**data)
        
        # Save report
        report_path = self.output_dir / f"stakeholder_report_{datetime.now().strftime('%Y%m%d')}.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        return report
    
    def _get_executive_template(self) -> str:
        """Executive summary template"""
        return """
# Energy Community Implementation - Executive Summary

**Date:** {{ date }}

## Key Performance Indicators

### 🎯 Overall Performance
- **Active Energy Communities:** {{ num_clusters }}
- **Buildings Participating:** {{ num_buildings }}
- **Average Self-Sufficiency:** {{ self_sufficiency }}
- **Total Cost Savings:** {{ cost_savings }} per month

### 🌱 Environmental Impact
- **CO2 Reduction:** {{ carbon_reduction }} tons/month
- **Peak Load Reduction:** {{ peak_reduction }}
- **Solar Coverage:** {{ solar_coverage }}

### 💰 Financial Metrics
- **Average ROI Period:** {{ roi_years }} years
- **Monthly Savings:** {{ cost_savings }}

{% if has_comparison %}
## Improvements Since Implementation
{% for metric, values in improvement.items() %}
- **{{ metric }}:** {{ values.before }} → {{ values.after }} ({{ values.improvement_percent }}% improvement)
{% endfor %}
{% endif %}

## Recommendations
1. Expand solar installations to buildings with poor energy labels (E/F/G)
2. Implement battery storage in high-demand clusters
3. Focus on increasing cluster complementarity through targeted recruitment

## Next Steps
- Review detailed technical report for implementation specifics
- Schedule stakeholder meetings for Phase 2 planning
- Approve budget for priority interventions
"""
    
    def _get_technical_template(self) -> str:
        """Technical report template"""
        return """
# Technical Analysis Report - Energy Community System

**Generated:** {{ date }}
**System Version:** 2.0

## System Performance Metrics

### Clustering Performance
- Number of Clusters: {{ metrics.num_clusters }}
- Average Cluster Size: {{ metrics.avg_cluster_size }}
- Cluster Stability: {{ metrics.cluster_stability }}%
- LV Boundary Compliance: 100%

### Energy Performance
- Total Demand: {{ metrics.total_demand_mwh }} MWh
- Total Generation: {{ metrics.total_generation_mwh }} MWh
- Grid Import: {{ metrics.grid_import_mwh }} MWh
- Grid Export: {{ metrics.grid_export_mwh }} MWh
- Line Losses: {{ metrics.line_loss_percent }}%

### Network Analysis
- Average Voltage Deviation: {{ metrics.avg_voltage_deviation }}%
- Transformer Utilization: {{ metrics.transformer_utilization_percent }}%
- Congestion Events: {{ metrics.congestion_events }}

## Detailed Cluster Analysis
{% for cluster_id, details in clusters.items() %}
### Cluster {{ cluster_id }}
- Quality Score: {{ details.quality_score }}
- Members: {{ details.member_count }}
- Self-Sufficiency: {{ details.self_sufficiency }}%
{% endfor %}

## Technical Recommendations
1. Upgrade transformer capacity in high-utilization areas
2. Implement advanced metering for real-time monitoring
3. Deploy grid-edge intelligence for autonomous operation

## System Logs
- Timestamp: {{ timestamp }}
- Processing Time: {{ network.processing_time }}ms
- Data Quality: {{ network.data_quality }}%
"""
    
    def _get_stakeholder_template(self) -> str:
        """Stakeholder-friendly report template"""
        return """
# Your Energy Community Report

**Date:** {{ date }}

## Your Community at a Glance

Welcome to your energy community! Here's how we're making a difference together:

### 👥 Community Size
Your community includes **{{ community_size }} buildings** working together to share clean energy and reduce costs.

### 💚 Environmental Impact
Together, we've saved **{{ carbon_saved }} tons of CO2** - equivalent to planting 🌳 {{ carbon_saved * 50 }} trees!

### 💰 Financial Benefits
- **Annual Savings:** {{ annual_savings }}
- **Green Energy Produced:** {{ green_energy }} MWh

## Benefits for You
{% for benefit in benefits %}
- {{ benefit }}
{% endfor %}

## What's Next?
{% for step in next_steps %}
{{ loop.index }}. {{ step }}
{% endfor %}

## How You Can Help
- Consider installing solar panels if you haven't already
- Shift flexible energy use to off-peak hours
- Participate in community energy events
- Spread the word to your neighbors

## Questions?
Contact your community energy manager or visit our online portal for more information.

---
*This report is automatically generated to keep you informed about your energy community's performance.*
"""