# main.py
import logging
from data.data_loader import DataLoader
from analysis.risk_metrics import RiskAnalyzer
from visualization.dashboards import DashboardCreator
import warnings
from datetime import datetime
import pandas as pd
import traceback

def setup_logging():
    """Configure logging with detailed formatting"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'risk_analytics_{datetime.now().strftime("%Y%m%d")}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def generate_risk_report(df, risk_metrics, logger):
    """Generate a summary report of the risk analysis"""
    try:
        # Get department metrics in a more accessible format
        dept_metrics = risk_metrics['departmental_metrics']['risk_metrics']['mean']
        dept_risks = pd.Series(dept_metrics, name='Risk Score')
        
        # Get regional metrics
        regional_metrics = risk_metrics['regional_metrics']['risk_metrics']['mean']
        region_risks = pd.Series(regional_metrics, name='Risk Score')
        
        report = f"""
Risk Analytics Summary Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

1. Overall Risk Metrics
----------------------
Total Risk Score: {risk_metrics['overall_metrics']['total_risk_score']:.2f}
Compliance Status: {risk_metrics['overall_metrics']['compliance_status']:.2f}%
High Risk Incidents: {risk_metrics['overall_metrics']['high_risk_incidents']}
Average Recovery Time: {risk_metrics['overall_metrics']['average_recovery_time']:.2f} hours

2. Departmental Risk Analysis
---------------------------
{dept_risks.to_string()}

3. Regional Risk Analysis
-----------------------
{region_risks.to_string()}

4. Trend Analysis
---------------
Risk Score Trends by Department:
{pd.Series(risk_metrics['trend_analysis']).to_string()}

5. Key Recommendations
--------------------
"""
        # Add recommendations if available
        if 'recommendations' in risk_metrics:
            for i, rec in enumerate(risk_metrics['recommendations'], 1):
                report += f"\n{i}. [{rec['priority']}] {rec['category']}: {rec['finding']}"
                report += f"\n   Action: {rec['action']}\n"
        
        # Save report to file
        report_filename = f'risk_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        with open(report_filename, 'w') as f:
            f.write(report)
        logger.info(f"Risk report generated successfully: {report_filename}")
        return report
    except Exception as e:
        logger.error(f"Error generating risk report: {str(e)}")
        logger.error(f"Stacktrace: {traceback.format_exc()}")
        return None

def main():
    # Filter warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting Enhanced Risk Analytics Pipeline")
    
    try:
        # Initialize components
        logger.info("Initializing components...")
        data_loader = DataLoader()
        risk_analyzer = RiskAnalyzer()
        dashboard_creator = DashboardCreator()
        
        # Load and process data
        logger.info("Loading risk data...")
        df = data_loader.load_sample_risk_data()
        logger.info(f"Loaded {len(df)} records of risk data")
        
        # Perform risk analysis
        logger.info("Performing comprehensive risk analysis...")
        df = risk_analyzer.identify_risk_clusters(df)
        risk_metrics = risk_analyzer.calculate_risk_metrics(df)
        
        # Generate recommendations
        logger.info("Generating risk recommendations...")
        recommendations = risk_analyzer.generate_risk_recommendations(risk_metrics)
        risk_metrics['recommendations'] = recommendations
        
        # Generate risk report
        logger.info("Generating risk analysis report...")
        report = generate_risk_report(df, risk_metrics, logger)
        
        # Create and launch dashboard
        logger.info("Creating interactive dashboard...")
        dashboard_creator.create_risk_dashboard(df, risk_metrics)
        
        # Configuration for dashboard
        host = '0.0.0.0'  # Allow external connections
        port = 8050       # Default Dash port
        
        logger.info(f"Dashboard ready - starting server on {host}:{port}")
        logger.info("Access the dashboard at http://127.0.0.1:8050")
        
        # Run the dashboard
        dashboard_creator.app.run_server(
            debug=True,
            host=host,
            port=port,
            dev_tools_hot_reload=True
        )
        
    except Exception as e:
        logger.error(f"Critical error in risk analytics pipeline: {str(e)}")
        logger.error(f"Stacktrace: {traceback.format_exc()}")
        raise
    finally:
        logger.info("Risk analytics pipeline completed")

if __name__ == "__main__":
    main()
