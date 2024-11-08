# analysis/risk_metrics.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy import stats

class RiskAnalyzer:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def calculate_risk_metrics(self, df):
        """Calculate comprehensive risk metrics"""
        metrics = {
            'overall_metrics': {
                'total_risk_score': df['risk_score'].mean(),
                'risk_volatility': df['risk_score'].std(),
                'high_risk_incidents': len(df[df['risk_score'] > 80]),
                'compliance_status': df['compliance_score'].mean(),
                'total_mitigation_cost': df['mitigation_cost'].sum(),
                'average_recovery_time': df['recovery_time_hours'].mean()
            },
            'departmental_metrics': self._calculate_department_metrics(df),
            'regional_metrics': self._calculate_regional_metrics(df),
            'trend_analysis': self._perform_trend_analysis(df),
            'risk_correlations': self._analyze_risk_correlations(df)
        }
        return metrics
    
    def _calculate_department_metrics(self, df):
        """Calculate department-level metrics"""
        dept_metrics = {}
        grouped = df.groupby('department')
        
        dept_metrics['risk_metrics'] = {
            'mean': grouped['risk_score'].mean().to_dict(),
            'max': grouped['risk_score'].max().to_dict(),
            'std': grouped['risk_score'].std().to_dict()
        }
        
        dept_metrics['incident_count'] = grouped['incident_count'].sum().to_dict()
        dept_metrics['compliance_score'] = grouped['compliance_score'].mean().to_dict()
        dept_metrics['budget_allocation'] = grouped['budget_allocation'].sum().to_dict()
        
        return dept_metrics
    
    def _calculate_regional_metrics(self, df):
        """Calculate region-level metrics"""
        regional_metrics = {}
        grouped = df.groupby('region')
        
        regional_metrics['risk_metrics'] = {
            'mean': grouped['risk_score'].mean().to_dict(),
            'max': grouped['risk_score'].max().to_dict()
        }
        
        regional_metrics['incident_count'] = grouped['incident_count'].sum().to_dict()
        regional_metrics['mitigation_cost'] = grouped['mitigation_cost'].sum().to_dict()
        
        return regional_metrics
    
    def _perform_trend_analysis(self, df):
        """Analyze trends in risk scores"""
        monthly_trends = df.groupby([df['date'].dt.to_period('M'), 'department'])['risk_score'].mean().unstack()
        trend_coefficients = {}
        for col in monthly_trends.columns:
            slope, _, _, _, _ = stats.linregress(range(len(monthly_trends)), monthly_trends[col])
            trend_coefficients[col] = slope
        return trend_coefficients
    
    def _analyze_risk_correlations(self, df):
        """Analyze correlations between risk metrics"""
        correlation_matrix = df[['risk_score', 'compliance_score', 'incident_count', 
                               'mitigation_cost', 'control_effectiveness']].corr()
        return correlation_matrix.to_dict()
    
    def identify_risk_clusters(self, df):
        """Identify risk clusters using PCA and KMeans"""
        features = ['risk_score', 'incident_count', 'compliance_score', 
                   'mitigation_cost', 'control_effectiveness']
        X = self.scaler.fit_transform(df[features])
        
        # Perform PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        df['risk_cluster'] = kmeans.fit_predict(X)
        df['pca_1'] = X_pca[:, 0]
        df['pca_2'] = X_pca[:, 1]
        
        return df
    
    def generate_risk_recommendations(self, metrics):
        """Generate risk recommendations based on metrics"""
        recommendations = []
        
        # High-risk departments
        dept_risks = metrics['departmental_metrics']['risk_metrics']['mean']
        high_risk_depts = [dept for dept, risk in dept_risks.items() if risk > 75]
        
        if high_risk_depts:
            recommendations.append({
                'priority': 'High',
                'category': 'Department Risk',
                'finding': f"High risk levels in departments: {', '.join(high_risk_depts)}",
                'action': "Immediate risk assessment and mitigation planning required"
            })
        
        # Compliance recommendations
        if metrics['overall_metrics']['compliance_status'] < 90:
            recommendations.append({
                'priority': 'Medium',
                'category': 'Compliance',
                'finding': "Below-target compliance scores",
                'action': "Review and enhance compliance training and monitoring"
            })
        
        # Cost optimization
        total_budget = sum(metrics['departmental_metrics']['budget_allocation'].values())
        total_mitigation_cost = metrics['overall_metrics']['total_mitigation_cost']
        
        if total_budget > 0:  # Prevent division by zero
            cost_ratio = total_mitigation_cost / total_budget
            if cost_ratio > 0.15:
                recommendations.append({
                    'priority': 'Medium',
                    'category': 'Cost Management',
                    'finding': "High mitigation costs relative to budget",
                    'action': "Review cost-effectiveness of current risk mitigation strategies"
                })
        
        # Add recommendations based on trends
        trend_analysis = metrics['trend_analysis']
        increasing_risk_depts = [dept for dept, slope in trend_analysis.items() if slope > 0.5]
        
        if increasing_risk_depts:
            recommendations.append({
                'priority': 'High',
                'category': 'Trend Analysis',
                'finding': f"Increasing risk trends in: {', '.join(increasing_risk_depts)}",
                'action': "Investigate root causes and implement preventive measures"
            })
        
        return recommendations
