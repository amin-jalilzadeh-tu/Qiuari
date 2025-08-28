# utils/metrics_tracker.py
"""
Performance monitoring and metrics tracking
Tracks model performance, system metrics, and energy KPIs
"""

import time
import psutil
import GPUtil
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from collections import defaultdict, deque
import json
import sqlite3
from pathlib import Path
import logging
import warnings

logger = logging.getLogger(__name__)

class MetricsTracker:
    """Track and store system and model metrics"""
    
    def __init__(self, 
                 db_path: str = "metrics.db",
                 buffer_size: int = 1000,
                 flush_interval: int = 60):
        """
        Initialize metrics tracker
        
        Args:
            db_path: Path to metrics database
            buffer_size: Size of metrics buffer
            flush_interval: Seconds between auto-flush
        """
        self.db_path = db_path
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        
        # Metrics buffer
        self.buffer = defaultdict(list)
        self.last_flush = time.time()
        
        # Performance counters
        self.counters = defaultdict(int)
        self.timers = defaultdict(list)
        
        # Moving averages
        self.window_size = 100
        self.moving_averages = defaultdict(lambda: deque(maxlen=self.window_size))
        
        # Initialize database
        self._init_database()
        
        # System monitors
        self.cpu_percent = 0
        self.memory_percent = 0
        self.gpu_percent = 0
        
        logger.info(f"Initialized MetricsTracker with database at {db_path}")
    
    def _init_database(self):
        """Initialize SQLite database for metrics storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                category TEXT,
                name TEXT,
                value REAL,
                metadata TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                event_type TEXT,
                description TEXT,
                metadata TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                task TEXT,
                metric_name TEXT,
                metric_value REAL,
                dataset TEXT,
                model_version TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def track_metric(self, category: str, name: str, value: float, 
                    metadata: Optional[Dict] = None):
        """
        Track a metric
        
        Args:
            category: Metric category
            name: Metric name
            value: Metric value
            metadata: Additional metadata
        """
        metric = {
            'timestamp': datetime.now(),
            'category': category,
            'name': name,
            'value': value,
            'metadata': json.dumps(metadata) if metadata else None
        }
        
        self.buffer['metrics'].append(metric)
        self.moving_averages[f"{category}.{name}"].append(value)
        
        # Auto-flush if needed
        if len(self.buffer['metrics']) >= self.buffer_size or \
           time.time() - self.last_flush > self.flush_interval:
            self.flush()
    
    def track_event(self, event_type: str, description: str, 
                   metadata: Optional[Dict] = None):
        """
        Track an event
        
        Args:
            event_type: Type of event
            description: Event description
            metadata: Additional metadata
        """
        event = {
            'timestamp': datetime.now(),
            'event_type': event_type,
            'description': description,
            'metadata': json.dumps(metadata) if metadata else None
        }
        
        self.buffer['events'].append(event)
        self.counters[event_type] += 1
    
    def track_model_performance(self, task: str, metrics: Dict, 
                               dataset: str = 'unknown',
                               model_version: str = 'unknown'):
        """
        Track model performance metrics
        
        Args:
            task: Task name
            metrics: Performance metrics
            dataset: Dataset name
            model_version: Model version
        """
        for metric_name, metric_value in metrics.items():
            perf = {
                'timestamp': datetime.now(),
                'task': task,
                'metric_name': metric_name,
                'metric_value': metric_value,
                'dataset': dataset,
                'model_version': model_version
            }
            
            self.buffer['model_performance'].append(perf)
    
    def start_timer(self, name: str) -> float:
        """
        Start a timer
        
        Args:
            name: Timer name
            
        Returns:
            Start time
        """
        start_time = time.time()
        self.timers[name].append(start_time)
        return start_time
    
    def stop_timer(self, name: str) -> float:
        """
        Stop a timer and record duration
        
        Args:
            name: Timer name
            
        Returns:
            Duration in seconds
        """
        if name in self.timers and self.timers[name]:
            start_time = self.timers[name].pop()
            duration = time.time() - start_time
            
            self.track_metric('timing', name, duration)
            
            return duration
        return 0.0
    
    def track_system_metrics(self):
        """Track system resource usage"""
        # CPU usage
        self.cpu_percent = psutil.cpu_percent(interval=1)
        self.track_metric('system', 'cpu_percent', self.cpu_percent)
        
        # Memory usage
        memory = psutil.virtual_memory()
        self.memory_percent = memory.percent
        self.track_metric('system', 'memory_percent', self.memory_percent)
        self.track_metric('system', 'memory_used_gb', memory.used / 1e9)
        
        # Disk usage
        disk = psutil.disk_usage('/')
        self.track_metric('system', 'disk_percent', disk.percent)
        
        # GPU usage (if available)
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                self.gpu_percent = gpu.load * 100
                self.track_metric('system', 'gpu_percent', self.gpu_percent)
                self.track_metric('system', 'gpu_memory_percent', gpu.memoryUtil * 100)
                self.track_metric('system', 'gpu_temperature', gpu.temperature)
        except:
            pass
        
        # Network I/O
        net_io = psutil.net_io_counters()
        self.track_metric('system', 'network_sent_mb', net_io.bytes_sent / 1e6)
        self.track_metric('system', 'network_recv_mb', net_io.bytes_recv / 1e6)
    
    def track_energy_metrics(self, energy_data: Dict):
        """
        Track energy-specific metrics
        
        Args:
            energy_data: Energy metrics dictionary
        """
        # Peak demand tracking
        if 'peak_demand' in energy_data:
            self.track_metric('energy', 'peak_demand_kw', energy_data['peak_demand'])
        
        # Renewable generation
        if 'solar_generation' in energy_data:
            self.track_metric('energy', 'solar_generation_kwh', energy_data['solar_generation'])
        
        # Grid metrics
        if 'grid_import' in energy_data:
            self.track_metric('energy', 'grid_import_kwh', energy_data['grid_import'])
        
        if 'grid_export' in energy_data:
            self.track_metric('energy', 'grid_export_kwh', energy_data['grid_export'])
        
        # Efficiency metrics
        if 'self_sufficiency' in energy_data:
            self.track_metric('energy', 'self_sufficiency', energy_data['self_sufficiency'])
        
        if 'self_consumption' in energy_data:
            self.track_metric('energy', 'self_consumption', energy_data['self_consumption'])
    
    def get_summary(self, category: Optional[str] = None, 
                   last_n_minutes: int = 60) -> Dict:
        """
        Get metrics summary
        
        Args:
            category: Filter by category
            last_n_minutes: Time window in minutes
            
        Returns:
            Summary dictionary
        """
        conn = sqlite3.connect(self.db_path)
        
        # Build query
        query = """
            SELECT category, name, 
                   AVG(value) as avg_value,
                   MIN(value) as min_value,
                   MAX(value) as max_value,
                   COUNT(*) as count
            FROM metrics
            WHERE timestamp > datetime('now', '-{} minutes')
        """.format(last_n_minutes)
        
        if category:
            query += f" AND category = '{category}'"
        
        query += " GROUP BY category, name"
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Convert to dictionary
        summary = {}
        for _, row in df.iterrows():
            key = f"{row['category']}.{row['name']}"
            summary[key] = {
                'avg': row['avg_value'],
                'min': row['min_value'],
                'max': row['max_value'],
                'count': row['count']
            }
        
        # Add moving averages
        for key, values in self.moving_averages.items():
            if values:
                summary[f"{key}_moving_avg"] = np.mean(values)
        
        # Add counters
        for key, count in self.counters.items():
            summary[f"count_{key}"] = count
        
        return summary
    
    def get_trends(self, metric_name: str, 
                  time_window: str = '1H') -> pd.DataFrame:
        """
        Get metric trends over time
        
        Args:
            metric_name: Metric name
            time_window: Time window for aggregation
            
        Returns:
            DataFrame with trends
        """
        conn = sqlite3.connect(self.db_path)
        
        query = f"""
            SELECT timestamp, value
            FROM metrics
            WHERE name = '{metric_name}'
            ORDER BY timestamp
        """
        
        df = pd.read_sql_query(query, conn, parse_dates=['timestamp'])
        conn.close()
        
        if not df.empty:
            # Resample to time window
            df.set_index('timestamp', inplace=True)
            df = df.resample(time_window).agg({
                'value': ['mean', 'min', 'max', 'std']
            })
        
        return df
    
    def flush(self):
        """Flush buffered metrics to database"""
        if not any(self.buffer.values()):
            return
        
        conn = sqlite3.connect(self.db_path)
        
        # Flush metrics
        if self.buffer['metrics']:
            df = pd.DataFrame(self.buffer['metrics'])
            df.to_sql('metrics', conn, if_exists='append', index=False)
            self.buffer['metrics'] = []
        
        # Flush events
        if self.buffer['events']:
            df = pd.DataFrame(self.buffer['events'])
            df.to_sql('events', conn, if_exists='append', index=False)
            self.buffer['events'] = []
        
        # Flush model performance
        if self.buffer['model_performance']:
            df = pd.DataFrame(self.buffer['model_performance'])
            df.to_sql('model_performance', conn, if_exists='append', index=False)
            self.buffer['model_performance'] = []
        
        conn.commit()
        conn.close()
        
        self.last_flush = time.time()
        logger.debug(f"Flushed metrics to database")
    
    def cleanup_old_metrics(self, days_to_keep: int = 30):
        """
        Remove old metrics from database
        
        Args:
            days_to_keep: Number of days to keep
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        for table in ['metrics', 'events', 'model_performance']:
            cursor.execute(f"""
                DELETE FROM {table}
                WHERE timestamp < ?
            """, (cutoff_date,))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Cleaned up metrics older than {days_to_keep} days")
    
    def export_metrics(self, output_path: str, format: str = 'csv'):
        """
        Export metrics to file
        
        Args:
            output_path: Output file path
            format: Export format ('csv', 'json', 'excel')
        """
        conn = sqlite3.connect(self.db_path)
        
        # Load all metrics
        metrics_df = pd.read_sql_query("SELECT * FROM metrics", conn)
        events_df = pd.read_sql_query("SELECT * FROM events", conn)
        perf_df = pd.read_sql_query("SELECT * FROM model_performance", conn)
        
        conn.close()
        
        if format == 'csv':
            metrics_df.to_csv(f"{output_path}_metrics.csv", index=False)
            events_df.to_csv(f"{output_path}_events.csv", index=False)
            perf_df.to_csv(f"{output_path}_performance.csv", index=False)
        elif format == 'json':
            metrics_df.to_json(f"{output_path}_metrics.json", orient='records')
            events_df.to_json(f"{output_path}_events.json", orient='records')
            perf_df.to_json(f"{output_path}_performance.json", orient='records')
        elif format == 'excel':
            with pd.ExcelWriter(f"{output_path}.xlsx") as writer:
                metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
                events_df.to_excel(writer, sheet_name='Events', index=False)
                perf_df.to_excel(writer, sheet_name='Performance', index=False)
        
        logger.info(f"Exported metrics to {output_path}")

class PerformanceMonitor:
    """Monitor and alert on performance issues"""
    
    def __init__(self, tracker: MetricsTracker):
        """
        Initialize performance monitor
        
        Args:
            tracker: MetricsTracker instance
        """
        self.tracker = tracker
        self.thresholds = {
            'cpu_percent': 80,
            'memory_percent': 85,
            'gpu_percent': 90,
            'response_time': 5.0,  # seconds
            'error_rate': 0.05  # 5%
        }
        
        self.alert_callbacks = []
    
    def add_alert_callback(self, callback: Callable):
        """Add callback for alerts"""
        self.alert_callbacks.append(callback)
    
    def check_thresholds(self):
        """Check if any thresholds are exceeded"""
        summary = self.tracker.get_summary(last_n_minutes=5)
        alerts = []
        
        # Check system metrics
        if 'system.cpu_percent' in summary:
            if summary['system.cpu_percent']['avg'] > self.thresholds['cpu_percent']:
                alerts.append({
                    'type': 'cpu_high',
                    'message': f"CPU usage high: {summary['system.cpu_percent']['avg']:.1f}%",
                    'severity': 'warning'
                })
        
        if 'system.memory_percent' in summary:
            if summary['system.memory_percent']['avg'] > self.thresholds['memory_percent']:
                alerts.append({
                    'type': 'memory_high',
                    'message': f"Memory usage high: {summary['system.memory_percent']['avg']:.1f}%",
                    'severity': 'warning'
                })
        
        # Check response times
        if 'timing.inference' in summary:
            if summary['timing.inference']['avg'] > self.thresholds['response_time']:
                alerts.append({
                    'type': 'slow_response',
                    'message': f"Slow inference: {summary['timing.inference']['avg']:.2f}s",
                    'severity': 'warning'
                })
        
        # Trigger alerts
        for alert in alerts:
            self._trigger_alert(alert)
        
        return alerts
    
    def _trigger_alert(self, alert: Dict):
        """Trigger alert callbacks"""
        logger.warning(f"Performance alert: {alert['message']}")
        
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

# Context manager for timing
class Timer:
    """Context manager for timing operations"""
    
    def __init__(self, tracker: MetricsTracker, name: str):
        self.tracker = tracker
        self.name = name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = self.tracker.start_timer(self.name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tracker.stop_timer(self.name)

# Usage example
if __name__ == "__main__":
    # Create tracker
    tracker = MetricsTracker(db_path="test_metrics.db")
    
    # Track some metrics
    tracker.track_metric('test', 'value1', 42.5)
    tracker.track_metric('test', 'value2', 13.7)
    
    # Track an event
    tracker.track_event('model_trained', 'Successfully trained clustering model')
    
    # Track model performance
    tracker.track_model_performance(
        'clustering',
        {'silhouette_score': 0.65, 'modularity': 0.72},
        dataset='delft_buildings',
        model_version='v1.2.0'
    )
    
    # Time an operation
    with Timer(tracker, 'test_operation'):
        time.sleep(0.5)
    
    # Track system metrics
    tracker.track_system_metrics()
    
    # Get summary
    summary = tracker.get_summary()
    print("Metrics Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Flush to database
    tracker.flush()
    
    # Create monitor
    monitor = PerformanceMonitor(tracker)
    alerts = monitor.check_thresholds()
    
    if alerts:
        print("\nAlerts:")
        for alert in alerts:
            print(f"  {alert['type']}: {alert['message']}")