# data/data_loader.py
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import logging
from datetime import datetime, timedelta

class DataLoader:
    def __init__(self, connection_string="sqlite:///risk_data.db"):
        self.connection_string = connection_string
        self.logger = logging.getLogger(__name__)
        
    def load_sample_risk_data(self):
        """Generate comprehensive risk data for demonstration"""
        np.random.seed(42)  # For reproducibility
        dates = pd.date_range(start='2023-01-01', periods=100)
        departments = ['IT', 'Finance', 'HR', 'Operations']
        regions = ['North', 'South', 'East', 'West']
        risk_types = ['Operational', 'Financial', 'Compliance', 'Strategic']
        
        # Generate more complex dataset
        data = []
        for date in dates:
            for dept in departments:
                for region in regions:
                    base_risk = np.random.uniform(40, 90)  # Base risk score
                    seasonal_factor = 10 * np.sin(2 * np.pi * date.month / 12)  # Seasonal pattern
                    
                    row = {
                        'date': date,
                        'department': dept,
                        'region': region,
                        'risk_score': min(100, max(0, base_risk + seasonal_factor)),
                        'incident_count': np.random.poisson(5),
                        'compliance_score': np.random.uniform(70, 100),
                        'budget_allocation': np.random.uniform(100000, 1000000),
                        'staff_count': np.random.randint(10, 100),
                        'risk_type': np.random.choice(risk_types),
                        'mitigation_cost': np.random.uniform(5000, 50000),
                        'previous_incidents': np.random.randint(0, 10),
                        'recovery_time_hours': np.random.exponential(24),
                        'control_effectiveness': np.random.uniform(60, 95)
                    }
                    data.append(row)
        
        return pd.DataFrame(data)
