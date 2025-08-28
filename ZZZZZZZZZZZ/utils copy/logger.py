# utils/logger.py
"""
Logging configuration for the energy GNN system
Provides structured logging with multiple handlers and formatters
"""

import logging
import logging.handlers
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import colorlog
import traceback

class StructuredFormatter(logging.Formatter):
    """JSON structured formatter for machine-readable logs"""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_obj = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields
        if hasattr(record, 'extra_fields'):
            log_obj.update(record.extra_fields)
        
        # Add exception info if present
        if record.exc_info:
            log_obj['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        return json.dumps(log_obj)

class ColoredFormatter(colorlog.ColoredFormatter):
    """Colored formatter for console output"""
    
    def __init__(self):
        super().__init__(
            fmt='%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s%(reset)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )

class LoggerConfig:
    """Logger configuration manager"""
    
    def __init__(self, 
                 name: str = 'energy_gnn',
                 level: str = 'INFO',
                 log_dir: str = 'logs',
                 console: bool = True,
                 file: bool = True,
                 structured: bool = False):
        """
        Initialize logger configuration
        
        Args:
            name: Logger name
            level: Logging level
            log_dir: Directory for log files
            console: Enable console logging
            file: Enable file logging
            structured: Use structured (JSON) logging
        """
        self.name = name
        self.level = getattr(logging, level.upper())
        self.log_dir = Path(log_dir)
        self.console = console
        self.file = file
        self.structured = structured
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logger
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup and configure logger"""
        logger = logging.getLogger(self.name)
        logger.setLevel(self.level)
        
        # Remove existing handlers
        logger.handlers.clear()
        
        # Console handler
        if self.console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.level)
            
            if self.structured:
                console_handler.setFormatter(StructuredFormatter())
            else:
                console_handler.setFormatter(ColoredFormatter())
            
            logger.addHandler(console_handler)
        
        # File handler
        if self.file:
            # Main log file
            file_handler = logging.handlers.RotatingFileHandler(
                self.log_dir / f'{self.name}.log',
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
            file_handler.setLevel(self.level)
            
            if self.structured:
                file_handler.setFormatter(StructuredFormatter())
            else:
                file_handler.setFormatter(logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                ))
            
            logger.addHandler(file_handler)
            
            # Error log file
            error_handler = logging.handlers.RotatingFileHandler(
                self.log_dir / f'{self.name}_errors.log',
                maxBytes=10*1024*1024,
                backupCount=5
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s\n%(exc_info)s'
            ))
            
            logger.addHandler(error_handler)
        
        # Prevent propagation to root logger
        logger.propagate = False
        
        return logger
    
    def get_logger(self, module_name: Optional[str] = None) -> logging.Logger:
        """
        Get logger instance
        
        Args:
            module_name: Module name for child logger
            
        Returns:
            Logger instance
        """
        if module_name:
            return logging.getLogger(f"{self.name}.{module_name}")
        return self.logger
    
    def set_level(self, level: str):
        """Change logging level"""
        self.level = getattr(logging, level.upper())
        self.logger.setLevel(self.level)
        for handler in self.logger.handlers:
            handler.setLevel(self.level)

class TaskLogger:
    """Logger for specific tasks with context"""
    
    def __init__(self, logger: logging.Logger, task: str):
        """
        Initialize task logger
        
        Args:
            logger: Base logger
            task: Task name
        """
        self.logger = logger
        self.task = task
        self.context = {'task': task}
        
    def log(self, level: str, message: str, **kwargs):
        """Log with task context"""
        extra = {'extra_fields': {**self.context, **kwargs}}
        getattr(self.logger, level)(message, extra=extra)
    
    def debug(self, message: str, **kwargs):
        self.log('debug', message, **kwargs)
    
    def info(self, message: str, **kwargs):
        self.log('info', message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        self.log('warning', message, **kwargs)
    
    def error(self, message: str, **kwargs):
        self.log('error', message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        self.log('critical', message, **kwargs)
    
    def log_metrics(self, metrics: Dict[str, Any]):
        """Log metrics"""
        self.info("Task metrics", **metrics)
    
    def log_start(self):
        """Log task start"""
        self.info(f"Starting {self.task} task")
    
    def log_complete(self, duration: float, success: bool = True):
        """Log task completion"""
        if success:
            self.info(f"Completed {self.task} task", duration=duration)
        else:
            self.error(f"Failed {self.task} task", duration=duration)

class ExperimentLogger:
    """Logger for ML experiments"""
    
    def __init__(self, 
                 experiment_name: str,
                 log_dir: str = 'experiments'):
        """
        Initialize experiment logger
        
        Args:
            experiment_name: Experiment name
            log_dir: Directory for experiment logs
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create experiment logger
        self.logger = logging.getLogger(f"experiment.{experiment_name}")
        self.logger.setLevel(logging.DEBUG)
        
        # File handler for experiment
        handler = logging.FileHandler(self.log_dir / 'experiment.log')
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)
        
        # Metrics file
        self.metrics_file = self.log_dir / 'metrics.json'
        self.metrics = []
        
        # Config file
        self.config_file = self.log_dir / 'config.json'
    
    def log_config(self, config: Dict):
        """Log experiment configuration"""
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
        self.logger.info(f"Logged configuration to {self.config_file}")
    
    def log_epoch(self, epoch: int, metrics: Dict):
        """Log epoch metrics"""
        entry = {
            'epoch': epoch,
            'timestamp': datetime.utcnow().isoformat(),
            **metrics
        }
        self.metrics.append(entry)
        
        # Append to file
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        # Log to logger
        self.logger.info(f"Epoch {epoch}: {metrics}")
    
    def log_best_model(self, epoch: int, metric_name: str, metric_value: float):
        """Log best model checkpoint"""
        self.logger.info(
            f"New best model at epoch {epoch}: {metric_name}={metric_value:.4f}"
        )
        
        # Save best model info
        best_file = self.log_dir / 'best_model.json'
        with open(best_file, 'w') as f:
            json.dump({
                'epoch': epoch,
                'metric_name': metric_name,
                'metric_value': metric_value,
                'timestamp': datetime.utcnow().isoformat()
            }, f, indent=2)
    
    def log_summary(self):
        """Log experiment summary"""
        if not self.metrics:
            return
        
        # Calculate summary statistics
        df = pd.DataFrame(self.metrics)
        summary = {
            'total_epochs': len(self.metrics),
            'best_metrics': {}
        }
        
        for col in df.columns:
            if col not in ['epoch', 'timestamp']:
                summary['best_metrics'][col] = {
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std())
                }
        
        # Save summary
        summary_file = self.log_dir / 'summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Experiment summary: {summary}")

# Global logger setup
def setup_logging(config: Optional[Dict] = None) -> logging.Logger:
    """
    Setup global logging configuration
    
    Args:
        config: Logging configuration dictionary
        
    Returns:
        Root logger
    """
    if config is None:
        config = {
            'name': 'energy_gnn',
            'level': 'INFO',
            'log_dir': 'logs',
            'console': True,
            'file': True,
            'structured': False
        }
    
    # Filter out unsupported config keys
    supported_keys = {'name', 'level', 'log_dir', 'console', 'file', 'structured'}
    filtered_config = {k: v for k, v in config.items() if k in supported_keys}
    
    # Create logger config
    logger_config = LoggerConfig(**filtered_config)
    
    # Setup loggers for all modules
    modules = [
        'data', 'models', 'training', 'tasks', 
        'inference', 'utils', 'api'
    ]
    
    for module in modules:
        module_logger = logger_config.get_logger(module)
        module_logger.setLevel(logger_config.level)
    
    return logger_config.get_logger()

# Decorators for logging
def log_execution(logger: logging.Logger):
    """Decorator to log function execution"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.debug(f"Starting {func.__name__}")
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.debug(f"Completed {func.__name__} in {duration:.2f}s")
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"Failed {func.__name__} after {duration:.2f}s: {e}")
                raise
        
        return wrapper
    return decorator

def log_errors(logger: logging.Logger):
    """Decorator to log exceptions"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
                raise
        
        return wrapper
    return decorator

# Usage example
if __name__ == "__main__":
    import time
    import pandas as pd
    
    # Setup logging
    logger = setup_logging({
        'name': 'test',
        'level': 'DEBUG',
        'console': True,
        'file': True
    })
    
    # Basic logging
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    
    # Task logger
    task_logger = TaskLogger(logger, 'clustering')
    task_logger.log_start()
    task_logger.log_metrics({
        'num_clusters': 5,
        'modularity': 0.65,
        'silhouette_score': 0.58
    })
    task_logger.log_complete(duration=3.5, success=True)
    
    # Experiment logger
    exp_logger = ExperimentLogger('test_experiment')
    exp_logger.log_config({'model': 'GNN', 'epochs': 100})
    
    for epoch in range(5):
        exp_logger.log_epoch(epoch, {
            'loss': 0.5 - epoch * 0.05,
            'accuracy': 0.7 + epoch * 0.03
        })
    
    exp_logger.log_best_model(4, 'accuracy', 0.82)
    exp_logger.log_summary()
    
    print("Logging examples completed")