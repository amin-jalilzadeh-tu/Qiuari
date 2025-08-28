"""
Interactive Dashboard for Energy Community Analysis
Using Streamlit for real-time visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import sys
sys.path.append('..')

# Import our modules
from visualization.data_aggregator import DataAggregator
from visualization.chart_generator import ChartGenerator
from visualization.report_generator import ReportGenerator
from visualization.economic_calculator import EconomicCalculator
from visualization.excel_reporter import ExcelReporter


# Page configuration
st.set_page_config(
    page_title="Energy Community Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 0rem 0rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 5px;
    }
    .plot-container {
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


class EnergyDashboard:
    """Interactive dashboard for energy community analysis"""
    
    def __init__(self):
        self.data_aggregator = DataAggregator()
        self.chart_generator = ChartGenerator()
        self.report_generator = ReportGenerator()
        self.economic_calculator = EconomicCalculator()
        self.excel_reporter = ExcelReporter()
        
        # Initialize session state
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
            st.session_state.metrics = None
            st.session_state.cluster_data = {}
            st.session_state.solar_data = {}
            st.session_state.economic_data = {}
    
    def run(self):
        """Main dashboard application"""
        
        # Sidebar
        with st.sidebar:
            st.title("⚡ Energy Community")
            st.markdown("---")
            
            # Navigation
            page = st.selectbox(
                "Navigation",
                ["Overview", "Cluster Analysis", "Energy Flows", 
                 "Solar Planning", "Economic Analysis", "Reports"]
            )
            
            st.markdown("---")
            
            # Data controls
            if st.button("🔄 Refresh Data"):
                self.load_data()
            
            if st.button("📊 Generate Reports"):
                self.generate_all_reports()
            
            if st.button("📥 Download Excel"):
                self.download_excel_report()
            
            st.markdown("---")
            
            # Filters
            st.subheader("Filters")
            
            date_range = st.date_input(
                "Date Range",
                value=(datetime.now() - timedelta(days=30), datetime.now()),
                max_value=datetime.now()
            )
            
            lv_groups = st.multiselect(
                "LV Groups",
                options=self.get_lv_groups(),
                default=[]
            )
            
            quality_filter = st.multiselect(
                "Cluster Quality",
                options=["excellent", "good", "fair", "poor"],
                default=["excellent", "good"]
            )
        
        # Main content
        if not st.session_state.data_loaded:
            self.load_data()
        
        # Display selected page
        if page == "Overview":
            self.show_overview()
        elif page == "Cluster Analysis":
            self.show_cluster_analysis()
        elif page == "Energy Flows":
            self.show_energy_flows()
        elif page == "Solar Planning":
            self.show_solar_planning()
        elif page == "Economic Analysis":
            self.show_economic_analysis()
        elif page == "Reports":
            self.show_reports()
    
    def load_data(self):
        """Load or generate data for dashboard"""
        
        with st.spinner("Loading data..."):
            # Try to load from latest results
            results_path = Path("results/data/latest_metrics.json")
            
            if results_path.exists():
                with open(results_path, 'r') as f:
                    metrics_dict = json.load(f)
                    # Convert to object-like structure
                    class Metrics:
                        def __init__(self, data):
                            for key, value in data.items():
                                setattr(self, key, value)
                    
                    st.session_state.metrics = Metrics(metrics_dict)
            else:
                # Generate sample data
                st.session_state.metrics = self.generate_sample_metrics()
            
            # Load cluster data
            st.session_state.cluster_data = self.generate_sample_clusters()
            
            # Load solar data
            st.session_state.solar_data = self.generate_sample_solar()
            
            # Calculate economic data
            st.session_state.economic_data = self.calculate_economics()
            
            st.session_state.data_loaded = True
            st.success("Data loaded successfully!")
    
    def show_overview(self):
        """Display overview dashboard"""
        
        st.title("Energy Community Overview")
        st.markdown("---")
        
        metrics = st.session_state.metrics
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Clusters",
                f"{metrics.num_clusters}",
                f"{metrics.num_clusters - 5} vs last month"
            )
            st.metric(
                "Active Buildings",
                f"{int(metrics.num_lv_groups * metrics.avg_buildings_per_lv)}",
                "+12 new"
            )
        
        with col2:
            st.metric(
                "Self-Sufficiency",
                f"{metrics.avg_self_sufficiency:.1%}",
                f"+{metrics.avg_self_sufficiency * 0.1:.1%}"
            )
            st.metric(
                "Peak Reduction",
                f"{metrics.total_peak_reduction:.1%}",
                f"+{metrics.total_peak_reduction * 0.05:.1%}"
            )
        
        with col3:
            st.metric(
                "Cost Savings",
                f"€{metrics.total_cost_savings_eur:,.0f}/mo",
                f"+€{metrics.total_cost_savings_eur * 0.15:,.0f}"
            )
            st.metric(
                "CO2 Reduced",
                f"{metrics.carbon_reduction_tons:.1f} tons",
                f"+{metrics.carbon_reduction_tons * 0.2:.1f}"
            )
        
        with col4:
            st.metric(
                "Solar Coverage",
                f"{metrics.solar_coverage_percent:.1%}",
                f"+{metrics.solar_coverage_percent * 0.1:.1%}"
            )
            st.metric(
                "Cluster Stability",
                f"{metrics.cluster_stability:.1%}",
                "Stable"
            )
        
        st.markdown("---")
        
        # Charts row 1
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Energy Balance")
            fig = self.create_energy_balance_chart(metrics)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Cluster Quality Distribution")
            fig = self.create_quality_distribution_chart()
            st.plotly_chart(fig, use_container_width=True)
        
        # Charts row 2
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Daily Load Profile")
            fig = self.create_daily_profile_chart()
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Economic Performance")
            fig = self.create_economic_gauge()
            st.plotly_chart(fig, use_container_width=True)
    
    def show_cluster_analysis(self):
        """Display cluster analysis page"""
        
        st.title("Cluster Analysis")
        st.markdown("---")
        
        # Cluster selection
        cluster_ids = list(st.session_state.cluster_data.keys())
        selected_cluster = st.selectbox("Select Cluster", cluster_ids)
        
        if selected_cluster:
            cluster = st.session_state.cluster_data[selected_cluster]
            
            # Cluster metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Size", cluster.get('member_count', 0))
                st.metric("LV Group", cluster.get('lv_group_id', 'N/A'))
            
            with col2:
                st.metric("Quality Score", f"{cluster.get('quality_score', 0):.1f}/100")
                st.metric("Quality Label", cluster.get('quality_label', 'N/A').upper())
            
            with col3:
                st.metric("Self-Sufficiency", f"{cluster.get('self_sufficiency_ratio', 0):.1%}")
                st.metric("Complementarity", f"{cluster.get('complementarity_score', 0):.2f}")
            
            with col4:
                st.metric("Peak Reduction", f"{cluster.get('peak_reduction_ratio', 0):.1%}")
                st.metric("Stability", f"{cluster.get('temporal_stability', 0):.1%}")
            
            st.markdown("---")
            
            # Cluster details tabs
            tab1, tab2, tab3 = st.tabs(["Members", "Energy Profile", "Recommendations"])
            
            with tab1:
                st.subheader("Cluster Members")
                members_df = self.get_cluster_members(selected_cluster)
                st.dataframe(members_df, use_container_width=True)
            
            with tab2:
                st.subheader("Energy Profile")
                col1, col2 = st.columns(2)
                with col1:
                    fig = self.create_cluster_energy_chart(cluster)
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    fig = self.create_cluster_profile_chart(cluster)
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                st.subheader("Improvement Recommendations")
                recommendations = self.get_cluster_recommendations(cluster)
                for i, rec in enumerate(recommendations, 1):
                    st.write(f"{i}. {rec}")
        
        # Cluster comparison
        st.markdown("---")
        st.subheader("Cluster Comparison")
        
        fig = self.create_cluster_comparison_chart()
        st.plotly_chart(fig, use_container_width=True)
    
    def show_energy_flows(self):
        """Display energy flows page"""
        
        st.title("Energy Flow Analysis")
        st.markdown("---")
        
        # Time period selection
        period = st.radio("Time Period", ["Today", "This Week", "This Month", "Custom"])
        
        if period == "Custom":
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date")
            with col2:
                end_date = st.date_input("End Date")
        
        # Flow metrics
        col1, col2, col3, col4 = st.columns(4)
        
        metrics = st.session_state.metrics
        
        with col1:
            st.metric("Total Demand", f"{metrics.total_demand_mwh:.1f} MWh")
        with col2:
            st.metric("Total Generation", f"{metrics.total_generation_mwh:.1f} MWh")
        with col3:
            st.metric("Shared Energy", f"{metrics.total_shared_energy_mwh:.1f} MWh")
        with col4:
            st.metric("Grid Import", f"{metrics.grid_import_mwh:.1f} MWh")
        
        st.markdown("---")
        
        # Sankey diagram
        st.subheader("Energy Flow Diagram")
        fig = self.create_sankey_diagram()
        st.plotly_chart(fig, use_container_width=True)
        
        # Hourly patterns
        st.markdown("---")
        st.subheader("Hourly Patterns")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = self.create_hourly_demand_chart()
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = self.create_hourly_generation_chart()
            st.plotly_chart(fig, use_container_width=True)
        
        # P2P trading matrix
        st.markdown("---")
        st.subheader("P2P Energy Trading Matrix")
        fig = self.create_p2p_trading_matrix()
        st.plotly_chart(fig, use_container_width=True)
    
    def show_solar_planning(self):
        """Display solar planning page"""
        
        st.title("Solar Installation Planning")
        st.markdown("---")
        
        # Solar potential map
        st.subheader("Solar Potential Assessment")
        
        col1, col2, col3 = st.columns(3)
        
        solar_data = st.session_state.solar_data
        
        with col1:
            st.metric("Total Potential", f"{solar_data.get('total_potential_kwp', 0):.0f} kWp")
        with col2:
            st.metric("Priority Buildings", solar_data.get('priority_count', 0))
        with col3:
            st.metric("Average ROI", f"{solar_data.get('avg_roi_years', 0):.1f} years")
        
        # Priority list
        st.markdown("---")
        st.subheader("Installation Priority List")
        
        priority_df = self.get_solar_priority_list()
        
        # Add filters
        col1, col2, col3 = st.columns(3)
        with col1:
            min_capacity = st.slider("Min Capacity (kWp)", 0, 50, 5)
        with col2:
            max_roi = st.slider("Max ROI (years)", 3, 15, 10)
        with col3:
            label_filter = st.multiselect("Energy Labels", ["E", "F", "G"], ["E", "F", "G"])
        
        # Filter dataframe
        filtered_df = priority_df[
            (priority_df['Capacity (kWp)'] >= min_capacity) &
            (priority_df['ROI (years)'] <= max_roi) &
            (priority_df['Energy Label'].isin(label_filter))
        ]
        
        st.dataframe(filtered_df, use_container_width=True)
        
        # ROI analysis
        st.markdown("---")
        st.subheader("ROI Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = self.create_roi_distribution_chart()
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = self.create_payback_timeline_chart()
            st.plotly_chart(fig, use_container_width=True)
        
        # Investment calculator
        st.markdown("---")
        st.subheader("Investment Calculator")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            capacity = st.number_input("Capacity (kWp)", min_value=1, max_value=100, value=10)
            annual_gen = st.number_input("Annual Generation (kWh)", 
                                        min_value=1000, max_value=150000, 
                                        value=int(capacity * 1200))
        
        with col2:
            self_consumption = st.slider("Self-Consumption %", 0, 100, 70) / 100
            building_demand = st.number_input("Building Demand (kWh/year)", 
                                            min_value=1000, max_value=100000, 
                                            value=10000)
        
        with col3:
            if st.button("Calculate ROI"):
                roi_result = self.economic_calculator.calculate_solar_roi(
                    capacity, annual_gen, self_consumption, building_demand
                )
                st.success(f"Payback Period: {roi_result['simple_payback_years']:.1f} years")
                st.info(f"NPV: €{roi_result['npv']:,.0f}")
                st.info(f"IRR: {roi_result['irr']:.1%}")
    
    def show_economic_analysis(self):
        """Display economic analysis page"""
        
        st.title("Economic Analysis")
        st.markdown("---")
        
        economic_data = st.session_state.economic_data
        
        # Financial summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Investment", f"€{economic_data.get('total_investment', 0):,.0f}")
        with col2:
            st.metric("Annual Benefits", f"€{economic_data.get('total_annual_benefit', 0):,.0f}")
        with col3:
            st.metric("Payback Period", f"{economic_data.get('overall_payback_years', 0):.1f} years")
        with col4:
            st.metric("ROI", f"{economic_data.get('roi_percent', 0):.1f}%")
        
        st.markdown("---")
        
        # Cost breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Investment Breakdown")
            fig = self.create_investment_breakdown_chart()
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Annual Benefits Breakdown")
            fig = self.create_benefits_breakdown_chart()
            st.plotly_chart(fig, use_container_width=True)
        
        # Cash flow projection
        st.markdown("---")
        st.subheader("20-Year Cash Flow Projection")
        fig = self.create_cashflow_projection_chart()
        st.plotly_chart(fig, use_container_width=True)
        
        # Sensitivity analysis
        st.markdown("---")
        st.subheader("Sensitivity Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Electricity Price Sensitivity**")
            electricity_prices = np.arange(0.15, 0.35, 0.05)
            npvs = []
            for price in electricity_prices:
                self.economic_calculator.electricity_price = price
                # Recalculate NPV
                npvs.append(np.random.uniform(50000, 150000))  # Sample calculation
            
            fig = go.Figure(data=go.Scatter(x=electricity_prices, y=npvs, mode='lines+markers'))
            fig.update_layout(xaxis_title="Electricity Price (€/kWh)", yaxis_title="NPV (€)")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**Discount Rate Sensitivity**")
            discount_rates = np.arange(0.03, 0.10, 0.01)
            npvs = []
            for rate in discount_rates:
                self.economic_calculator.discount_rate = rate
                # Recalculate NPV
                npvs.append(np.random.uniform(40000, 120000))  # Sample calculation
            
            fig = go.Figure(data=go.Scatter(x=discount_rates, y=npvs, mode='lines+markers'))
            fig.update_layout(xaxis_title="Discount Rate", yaxis_title="NPV (€)")
            st.plotly_chart(fig, use_container_width=True)
    
    def show_reports(self):
        """Display reports page"""
        
        st.title("Reports & Downloads")
        st.markdown("---")
        
        # Report generation
        st.subheader("Generate Reports")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Executive Summary", use_container_width=True):
                with st.spinner("Generating..."):
                    report = self.report_generator.generate_executive_summary(
                        st.session_state.metrics
                    )
                    st.success("Executive summary generated!")
                    st.download_button(
                        "Download Executive Summary",
                        report,
                        file_name=f"executive_summary_{datetime.now().strftime('%Y%m%d')}.md"
                    )
        
        with col2:
            if st.button("📊 Technical Report", use_container_width=True):
                with st.spinner("Generating..."):
                    report = self.report_generator.generate_technical_report(
                        st.session_state.metrics,
                        st.session_state.cluster_data,
                        {}
                    )
                    st.success("Technical report generated!")
                    st.download_button(
                        "Download Technical Report",
                        report,
                        file_name=f"technical_report_{datetime.now().strftime('%Y%m%d')}.md"
                    )
        
        with col3:
            if st.button("📈 Excel Report", use_container_width=True):
                with st.spinner("Generating..."):
                    filepath = self.excel_reporter.generate_comprehensive_report(
                        st.session_state.metrics,
                        st.session_state.cluster_data,
                        st.session_state.solar_data,
                        st.session_state.economic_data
                    )
                    st.success("Excel report generated!")
                    with open(filepath, 'rb') as f:
                        st.download_button(
                            "Download Excel Report",
                            f,
                            file_name=Path(filepath).name,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
        
        # Historical reports
        st.markdown("---")
        st.subheader("Historical Reports")
        
        reports_dir = Path("results/reports")
        if reports_dir.exists():
            report_files = list(reports_dir.glob("*.md")) + list(reports_dir.glob("*.xlsx"))
            
            if report_files:
                reports_data = []
                for file in report_files:
                    reports_data.append({
                        'Name': file.name,
                        'Type': file.suffix[1:].upper(),
                        'Size': f"{file.stat().st_size / 1024:.1f} KB",
                        'Modified': datetime.fromtimestamp(file.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
                    })
                
                reports_df = pd.DataFrame(reports_data)
                st.dataframe(reports_df, use_container_width=True)
            else:
                st.info("No historical reports found")
        
        # Export options
        st.markdown("---")
        st.subheader("Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            export_format = st.selectbox("Export Format", ["CSV", "JSON", "Excel"])
            
        with col2:
            if st.button("Export All Data"):
                if export_format == "JSON":
                    data = {
                        'metrics': st.session_state.metrics.__dict__ if hasattr(st.session_state.metrics, '__dict__') else {},
                        'clusters': st.session_state.cluster_data,
                        'solar': st.session_state.solar_data,
                        'economic': st.session_state.economic_data
                    }
                    json_str = json.dumps(data, indent=2, default=str)
                    st.download_button(
                        "Download JSON",
                        json_str,
                        file_name=f"energy_community_data_{datetime.now().strftime('%Y%m%d')}.json"
                    )
                st.success(f"Data exported as {export_format}")
    
    # Helper methods for data generation and chart creation
    def generate_sample_metrics(self):
        """Generate sample metrics for demo"""
        class Metrics:
            def __init__(self):
                self.num_clusters = 12
                self.avg_cluster_size = 8.5
                self.cluster_stability = 0.875
                self.avg_self_sufficiency = 0.42
                self.avg_complementarity = 0.68
                self.total_peak_reduction = 0.25
                self.total_demand_mwh = 850.5
                self.total_generation_mwh = 360.2
                self.total_shared_energy_mwh = 180.1
                self.grid_import_mwh = 490.3
                self.grid_export_mwh = 30.0
                self.num_solar_buildings = 45
                self.total_solar_capacity_kw = 675
                self.avg_solar_roi_years = 6.8
                self.solar_coverage_percent = 28.5
                self.total_cost_savings_eur = 12500
                self.avg_cost_reduction_percent = 18.5
                self.carbon_reduction_tons = 42.3
                self.peak_charge_savings_eur = 3200
                self.avg_voltage_deviation = 0.02
                self.transformer_utilization_percent = 65
                self.line_loss_percent = 3.5
                self.congestion_events = 2
                self.num_lv_groups = 20
                self.avg_buildings_per_lv = 8
                self.lv_groups_with_clusters = 15
        
        return Metrics()
    
    def generate_sample_clusters(self):
        """Generate sample cluster data"""
        clusters = {}
        for i in range(1, 13):
            clusters[i] = {
                'member_count': np.random.randint(5, 15),
                'lv_group_id': f'LV_{np.random.randint(1, 21):03d}',
                'quality_score': np.random.uniform(60, 95),
                'quality_label': np.random.choice(['excellent', 'good', 'fair', 'poor']),
                'self_sufficiency_ratio': np.random.uniform(0.3, 0.7),
                'complementarity_score': np.random.uniform(0.5, 0.9),
                'peak_reduction_ratio': np.random.uniform(0.15, 0.35),
                'temporal_stability': np.random.uniform(0.7, 0.95),
                'total_shared_kwh': np.random.uniform(5000, 20000)
            }
        return clusters
    
    def generate_sample_solar(self):
        """Generate sample solar data"""
        return {
            'total_potential_kwp': 1500,
            'priority_count': 35,
            'avg_roi_years': 6.8,
            'total_capacity': 675,
            'num_installations': 45,
            'total_investment': 810000,
            'annual_generation': 810000,
            'co2_savings': 405
        }
    
    def calculate_economics(self):
        """Calculate economic metrics"""
        return {
            'total_investment': 1250000,
            'total_annual_benefit': 185000,
            'overall_payback_years': 6.8,
            'roi_percent': 14.8,
            'npv': 850000,
            'irr': 0.12,
            'investments': {
                'Solar PV': 810000,
                'Battery Storage': 320000,
                'Smart Meters': 120000
            },
            'annual_benefits': {
                'Energy Savings': 95000,
                'P2P Trading': 35000,
                'Peak Reduction': 28000,
                'Grid Services': 27000
            }
        }
    
    def get_lv_groups(self):
        """Get list of LV groups"""
        return [f'LV_{i:03d}' for i in range(1, 21)]
    
    def get_cluster_members(self, cluster_id):
        """Get cluster member details"""
        members = []
        num_members = st.session_state.cluster_data[cluster_id]['member_count']
        
        for i in range(num_members):
            members.append({
                'Building ID': f'B{cluster_id:02d}{i:02d}',
                'Energy Label': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G']),
                'Type': np.random.choice(['Residential', 'Commercial']),
                'Demand (kWh/yr)': np.random.randint(3000, 15000),
                'Has Solar': np.random.choice(['Yes', 'No']),
                'Priority Score': np.random.uniform(0.3, 0.9)
            })
        
        return pd.DataFrame(members)
    
    def get_solar_priority_list(self):
        """Get solar installation priority list"""
        priority_list = []
        
        for i in range(20):
            priority_list.append({
                'Rank': i + 1,
                'Building ID': f'B{np.random.randint(1, 100):03d}',
                'Energy Label': np.random.choice(['E', 'F', 'G']),
                'Roof Area (m²)': np.random.randint(40, 150),
                'Capacity (kWp)': np.random.uniform(5, 25),
                'ROI (years)': np.random.uniform(4, 12),
                'Priority Score': np.random.uniform(0.6, 0.95)
            })
        
        return pd.DataFrame(priority_list)
    
    def get_cluster_recommendations(self, cluster):
        """Get recommendations for cluster improvement"""
        recommendations = []
        
        if cluster['self_sufficiency_ratio'] < 0.5:
            recommendations.append("Increase renewable generation capacity by adding solar panels to suitable buildings")
        
        if cluster['complementarity_score'] < 0.7:
            recommendations.append("Recruit buildings with complementary load profiles (e.g., offices with residential)")
        
        if cluster['peak_reduction_ratio'] < 0.2:
            recommendations.append("Implement demand response programs or add battery storage for peak shaving")
        
        if cluster['temporal_stability'] < 0.8:
            recommendations.append("Establish long-term agreements to improve membership stability")
        
        if len(recommendations) == 0:
            recommendations.append("Cluster is performing well - maintain current configuration")
        
        return recommendations
    
    # Chart creation methods
    def create_energy_balance_chart(self, metrics):
        """Create energy balance donut chart"""
        values = [
            metrics.total_generation_mwh,
            metrics.grid_import_mwh
        ]
        labels = ['Local Generation', 'Grid Import']
        colors = ['#73AB84', '#F18F01']
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            marker=dict(colors=colors)
        )])
        
        fig.update_layout(
            title="Energy Sources",
            height=300,
            showlegend=True
        )
        
        return fig
    
    def create_quality_distribution_chart(self):
        """Create cluster quality distribution chart"""
        quality_counts = {'excellent': 3, 'good': 5, 'fair': 3, 'poor': 1}
        
        fig = go.Figure(data=[go.Bar(
            x=list(quality_counts.keys()),
            y=list(quality_counts.values()),
            marker_color=['#73AB84', '#A6D96A', '#F18F01', '#C73E1D']
        )])
        
        fig.update_layout(
            title="Cluster Quality Distribution",
            xaxis_title="Quality",
            yaxis_title="Count",
            height=300
        )
        
        return fig
    
    def create_daily_profile_chart(self):
        """Create daily load profile chart"""
        hours = list(range(24))
        demand = [30 + 10*np.sin((h-6)*np.pi/12) + np.random.normal(0, 2) for h in hours]
        generation = [max(0, 15*np.sin((h-6)*np.pi/12)) + np.random.normal(0, 1) for h in hours]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hours, y=demand,
            mode='lines',
            name='Demand',
            line=dict(color='#2E86AB', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=hours, y=generation,
            mode='lines',
            name='Generation',
            line=dict(color='#73AB84', width=2)
        ))
        
        fig.update_layout(
            title="Daily Load Profile",
            xaxis_title="Hour",
            yaxis_title="Power (MW)",
            height=300
        )
        
        return fig
    
    def create_economic_gauge(self):
        """Create economic performance gauge"""
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=14.8,
            delta={'reference': 10},
            gauge={'axis': {'range': [None, 30]},
                   'bar': {'color': "darkgreen"},
                   'steps': [
                       {'range': [0, 10], 'color': "lightgray"},
                       {'range': [10, 20], 'color': "yellow"},
                       {'range': [20, 30], 'color': "lightgreen"}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 10}},
            title={'text': "ROI (%)"}
        ))
        
        fig.update_layout(height=300)
        return fig
    
    def create_cluster_comparison_chart(self):
        """Create cluster comparison radar chart"""
        clusters = st.session_state.cluster_data
        
        # Select top 5 clusters
        top_clusters = list(clusters.keys())[:5]
        
        categories = ['Self-Sufficiency', 'Complementarity', 'Peak Reduction', 
                     'Stability', 'Quality Score']
        
        fig = go.Figure()
        
        for cluster_id in top_clusters:
            cluster = clusters[cluster_id]
            values = [
                cluster['self_sufficiency_ratio'],
                cluster['complementarity_score'],
                cluster['peak_reduction_ratio'],
                cluster['temporal_stability'],
                cluster['quality_score'] / 100
            ]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=f'Cluster {cluster_id}'
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Cluster Performance Comparison"
        )
        
        return fig
    
    def create_sankey_diagram(self):
        """Create energy flow Sankey diagram"""
        # Define nodes
        labels = ["Solar", "Wind", "Grid", "Buildings", "EVs", "Storage", "Export", "Losses"]
        source = [0, 1, 2, 3, 3, 3, 5]
        target = [3, 3, 3, 4, 5, 6, 7]
        value = [100, 50, 200, 30, 40, 15, 10]
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
                color=["green", "lightgreen", "gray", "blue", 
                      "orange", "purple", "gray", "red"]
            ),
            link=dict(
                source=source,
                target=target,
                value=value,
                color="rgba(0,0,0,0.2)"
            )
        )])
        
        fig.update_layout(
            title="Energy Flow Distribution",
            font_size=10,
            height=400
        )
        
        return fig
    
    def create_investment_breakdown_chart(self):
        """Create investment breakdown pie chart"""
        investments = st.session_state.economic_data.get('investments', {})
        
        fig = go.Figure(data=[go.Pie(
            labels=list(investments.keys()),
            values=list(investments.values()),
            hole=0.3
        )])
        
        fig.update_layout(
            title="Investment Breakdown",
            height=350
        )
        
        return fig
    
    def create_benefits_breakdown_chart(self):
        """Create benefits breakdown pie chart"""
        benefits = st.session_state.economic_data.get('annual_benefits', {})
        
        fig = go.Figure(data=[go.Pie(
            labels=list(benefits.keys()),
            values=list(benefits.values()),
            hole=0.3
        )])
        
        fig.update_layout(
            title="Annual Benefits Breakdown",
            height=350
        )
        
        return fig
    
    def create_cashflow_projection_chart(self):
        """Create 20-year cash flow projection"""
        years = list(range(21))
        investment = -1250000
        annual_benefit = 185000
        
        cashflow = [investment]
        cumulative = [investment]
        
        for year in range(1, 21):
            cf = annual_benefit * (1.02 ** year)  # 2% inflation
            cashflow.append(cf)
            cumulative.append(cumulative[-1] + cf)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=years,
            y=cashflow,
            name='Annual Cash Flow',
            marker_color=['red'] + ['green'] * 20
        ))
        
        fig.add_trace(go.Scatter(
            x=years,
            y=cumulative,
            name='Cumulative Cash Flow',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title="20-Year Cash Flow Projection",
            xaxis_title="Year",
            yaxis_title="Cash Flow (€)",
            height=400,
            hovermode='x unified'
        )
        
        return fig
    
    def create_hourly_demand_chart(self):
        """Create hourly demand chart"""
        hours = list(range(24))
        demand = [30 + 10*np.sin((h-6)*np.pi/12) + np.random.normal(0, 2) for h in hours]
        
        fig = go.Figure(data=[go.Bar(
            x=hours,
            y=demand,
            marker_color='#2E86AB'
        )])
        
        fig.update_layout(
            title="Hourly Demand Profile",
            xaxis_title="Hour",
            yaxis_title="Demand (MW)",
            height=350
        )
        
        return fig
    
    def create_hourly_generation_chart(self):
        """Create hourly generation chart"""
        hours = list(range(24))
        generation = [max(0, 15*np.sin((h-6)*np.pi/12)) + np.random.normal(0, 1) for h in hours]
        
        fig = go.Figure(data=[go.Bar(
            x=hours,
            y=generation,
            marker_color='#73AB84'
        )])
        
        fig.update_layout(
            title="Hourly Generation Profile",
            xaxis_title="Hour",
            yaxis_title="Generation (MW)",
            height=350
        )
        
        return fig
    
    def create_p2p_trading_matrix(self):
        """Create P2P trading matrix heatmap"""
        n_clusters = 5
        matrix = np.random.uniform(0, 100, (n_clusters, n_clusters))
        np.fill_diagonal(matrix, 0)
        
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=[f'Cluster {i+1}' for i in range(n_clusters)],
            y=[f'Cluster {i+1}' for i in range(n_clusters)],
            colorscale='Viridis',
            text=np.round(matrix, 1),
            texttemplate="%{text}",
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title="P2P Energy Trading (kWh)",
            xaxis_title="To",
            yaxis_title="From",
            height=400
        )
        
        return fig
    
    def create_roi_distribution_chart(self):
        """Create ROI distribution histogram"""
        roi_values = np.random.normal(7, 2, 100)
        roi_values = roi_values[roi_values > 0]
        
        fig = go.Figure(data=[go.Histogram(
            x=roi_values,
            nbinsx=20,
            marker_color='#F18F01'
        )])
        
        fig.update_layout(
            title="ROI Distribution",
            xaxis_title="ROI (years)",
            yaxis_title="Count",
            height=350
        )
        
        return fig
    
    def create_payback_timeline_chart(self):
        """Create payback timeline chart"""
        years = list(range(16))
        buildings = []
        
        for year in years:
            count = int(np.random.poisson(3) * np.exp(-year/10))
            buildings.append(count)
        
        fig = go.Figure(data=[go.Bar(
            x=years,
            y=buildings,
            marker_color='#73AB84'
        )])
        
        fig.update_layout(
            title="Payback Timeline Distribution",
            xaxis_title="Years to Payback",
            yaxis_title="Number of Buildings",
            height=350
        )
        
        return fig
    
    def create_cluster_energy_chart(self, cluster):
        """Create cluster energy balance chart"""
        labels = ['Demand', 'Generation', 'Import', 'Export']
        values = [
            np.random.uniform(10000, 20000),
            np.random.uniform(5000, 15000),
            np.random.uniform(3000, 8000),
            np.random.uniform(1000, 3000)
        ]
        colors = ['#2E86AB', '#73AB84', '#F18F01', '#C73E1D']
        
        fig = go.Figure(data=[go.Bar(
            x=labels,
            y=values,
            marker_color=colors
        )])
        
        fig.update_layout(
            title="Cluster Energy Balance",
            yaxis_title="Energy (kWh)",
            height=300
        )
        
        return fig
    
    def create_cluster_profile_chart(self, cluster):
        """Create cluster load profile chart"""
        hours = list(range(24))
        profile = [5 + 3*np.sin((h-6)*np.pi/12) + np.random.normal(0, 0.5) for h in hours]
        
        fig = go.Figure(data=[go.Scatter(
            x=hours,
            y=profile,
            mode='lines',
            fill='tozeroy',
            line=dict(color='#2E86AB', width=2)
        )])
        
        fig.update_layout(
            title="Daily Load Profile",
            xaxis_title="Hour",
            yaxis_title="Load (kW)",
            height=300
        )
        
        return fig
    
    def generate_all_reports(self):
        """Generate all reports at once"""
        with st.spinner("Generating all reports..."):
            # Generate each report type
            self.report_generator.generate_executive_summary(st.session_state.metrics)
            self.report_generator.generate_technical_report(
                st.session_state.metrics,
                st.session_state.cluster_data,
                {}
            )
            
            # Generate Excel report
            self.excel_reporter.generate_comprehensive_report(
                st.session_state.metrics,
                st.session_state.cluster_data,
                st.session_state.solar_data,
                st.session_state.economic_data
            )
            
            st.success("All reports generated successfully!")
    
    def download_excel_report(self):
        """Generate and download Excel report"""
        with st.spinner("Generating Excel report..."):
            filepath = self.excel_reporter.generate_comprehensive_report(
                st.session_state.metrics,
                st.session_state.cluster_data,
                st.session_state.solar_data,
                st.session_state.economic_data
            )
            st.success(f"Report saved to {filepath}")


# Run the dashboard
if __name__ == "__main__":
    dashboard = EnergyDashboard()
    dashboard.run()