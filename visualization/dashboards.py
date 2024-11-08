# visualization/dashboards.py
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
from datetime import datetime

class DashboardCreator:
    def __init__(self):
        self.app = Dash(__name__)
        self.color_scales = {
            'risk': px.colors.sequential.Reds,
            'compliance': px.colors.sequential.Blues,
            'general': px.colors.qualitative.Set3
        }
        
    def _create_metric_card(self, title, value, indicator, details=None):
        """Create a metric card with title, value, and trend indicator"""
        indicator_color = {
            "↑": "text-green-500" if title != "Risk Score" else "text-red-500",
            "↓": "text-red-500" if title != "Risk Score" else "text-green-500",
            "⚠️": "text-yellow-500"
        }.get(indicator, "text-gray-500")
        
        return html.Div([
            html.H3(title, className="text-lg font-semibold text-gray-700"),
            html.Div([
                html.Span(
                    f"{value:.1f}" if isinstance(value, float) else value,
                    className="text-3xl font-bold"
                ),
                html.Span(indicator, className=f"ml-2 text-xl {indicator_color}")
            ], className="flex items-center"),
            html.P(details or "", className="text-sm text-gray-500 mt-2")
        ], className="p-4 bg-white shadow-lg rounded-lg")

    def create_risk_dashboard(self, df, risk_metrics):
        """Create comprehensive risk management dashboard"""
        self.df = df
        self.risk_metrics = risk_metrics
        
        self.app.layout = html.Div([
            # Header Section
            self._create_header(),
            
            # Metrics Overview Section
            self._create_metrics_overview(),
            
            # Filters Section
            self._create_filters(),
            
            # Main Content Grid
            html.Div([
                # Left Column
                html.Div([
                    self._create_risk_trends_panel(),
                    self._create_department_comparison_panel(),
                ], className="col-span-2"),
                
                # Right Column
                html.Div([
                    self._create_risk_distribution_panel(),
                    self._create_compliance_panel(),
                ], className="col-span-1"),
            ], className="grid grid-cols-3 gap-4 p-4"),
            
            # Bottom Section
            html.Div([
                self._create_risk_cluster_panel(),
                self._create_recommendations_panel(),
            ], className="grid grid-cols-2 gap-4 p-4"),
            
            # Footer
            self._create_footer()
            
        ], className="bg-gray-100 min-h-screen")
        
        self._setup_callbacks()

    def _create_header(self):
        """Create dashboard header"""
        return html.Div([
            html.Div([
                html.H1("Enterprise Risk Management Dashboard", 
                        className="text-3xl font-bold mb-2"),
                html.P(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                      className="text-sm text-gray-500")
            ], className="flex justify-between items-center"),
            html.Hr(className="my-4")
        ], className="p-4 bg-white shadow-lg")

    def _create_metrics_overview(self):
        """Create metrics overview section"""
        return html.Div([
            self._create_metric_card(
                "Overall Risk Score",
                self.risk_metrics['overall_metrics']['total_risk_score'],
                "↑" if self.risk_metrics['overall_metrics']['total_risk_score'] > 75 else "↓",
                f"Volatility: {self.risk_metrics['overall_metrics']['risk_volatility']:.2f}"
            ),
            self._create_metric_card(
                "Compliance Rate",
                self.risk_metrics['overall_metrics']['compliance_status'],
                "↑" if self.risk_metrics['overall_metrics']['compliance_status'] > 90 else "↓",
                "Target: 90%"
            ),
            self._create_metric_card(
                "High Risk Incidents",
                self.risk_metrics['overall_metrics']['high_risk_incidents'],
                "⚠️",
                "Incidents with risk score > 80"
            ),
            self._create_metric_card(
                "Avg Recovery Time",
                self.risk_metrics['overall_metrics']['average_recovery_time'],
                "↔",
                "Hours to resolve incidents"
            )
        ], className="grid grid-cols-4 gap-4 p-4")

    def _create_filters(self):
        """Create filters section"""
        return html.Div([
            html.Div([
                html.Label("Department", className="block text-sm font-medium text-gray-700"),
                dcc.Dropdown(
                    id='department-filter',
                    options=[{'label': d, 'value': d} for d in self.df['department'].unique()],
                    value='All',
                    className="mt-1"
                ),
            ], className="w-1/4"),
            
            html.Div([
                html.Label("Date Range", className="block text-sm font-medium text-gray-700"),
                dcc.DatePickerRange(
                    id='date-range',
                    start_date=self.df['date'].min(),
                    end_date=self.df['date'].max(),
                    className="mt-1"
                ),
            ], className="w-1/2"),
            
            html.Div([
                html.Label("Risk Threshold", className="block text-sm font-medium text-gray-700"),
                dcc.Slider(
                    id='risk-threshold',
                    min=0,
                    max=100,
                    value=80,
                    marks={i: str(i) for i in range(0, 101, 20)},
                    className="mt-1"
                ),
            ], className="w-1/4"),
        ], className="flex justify-between items-end p-4 bg-white shadow-lg")

    def _create_risk_trends_panel(self):
        """Create risk trends visualization panel"""
        return html.Div([
            html.H2("Risk Score Trends", className="text-xl font-bold mb-4"),
            dcc.Graph(id='risk-trend-plot')
        ], className="p-4 bg-white shadow-lg")

    def _create_department_comparison_panel(self):
        """Create department comparison panel"""
        return html.Div([
            html.H2("Department Risk Comparison", className="text-xl font-bold mb-4"),
            dcc.Graph(id='department-comparison-plot')
        ], className="p-4 bg-white shadow-lg")

    def _create_risk_distribution_panel(self):
        """Create risk distribution panel"""
        return html.Div([
            html.H2("Risk Distribution", className="text-xl font-bold mb-4"),
            dcc.Graph(id='risk-distribution-plot')
        ], className="p-4 bg-white shadow-lg")

    def _create_compliance_panel(self):
        """Create compliance panel"""
        return html.Div([
            html.H2("Compliance Overview", className="text-xl font-bold mb-4"),
            dcc.Graph(id='compliance-plot')
        ], className="p-4 bg-white shadow-lg")

    def _create_risk_cluster_panel(self):
        """Create risk cluster visualization panel"""
        return html.Div([
            html.H2("Risk Clusters Analysis", className="text-xl font-bold mb-4"),
            dcc.Graph(id='risk-cluster-plot')
        ], className="p-4 bg-white shadow-lg")

    def _create_recommendations_panel(self):
        """Create recommendations panel"""
        return html.Div([
            html.H2("Risk Mitigation Recommendations", className="text-xl font-bold mb-4"),
            html.Div(
                self._create_recommendations_table(),
                className="overflow-x-auto"
            )
        ], className="p-4 bg-white shadow-lg")

    def _create_recommendations_table(self):
        """Create recommendations table"""
        return html.Table(
            [
                # Header
                html.Thead(
                    html.Tr([
                        html.Th(col, className="px-4 py-2 text-left text-sm font-semibold text-gray-900")
                        for col in ['Priority', 'Category', 'Finding', 'Action']
                    ])
                ),
                # Body
                html.Tbody([
                    html.Tr([
                        html.Td(rec[col], className="px-4 py-2 text-sm text-gray-900")
                        for col in ['priority', 'category', 'finding', 'action']
                    ], className=f"{'bg-gray-50' if i % 2 else ''}")
                    for i, rec in enumerate(self.risk_metrics.get('recommendations', []))
                ])
            ],
            className="min-w-full divide-y divide-gray-300"
        )

    def _create_footer(self):
        """Create dashboard footer"""
        return html.Footer([
            html.P(
                "Enterprise Risk Management Dashboard - Updated automatically",
                className="text-center text-sm text-gray-500"
            )
        ], className="p-4 mt-8")

    def _create_risk_trend_plot(self, df):
        """Create risk trend visualization"""
        fig = px.line(df,
                     x='date',
                     y='risk_score',
                     color='department',
                     title='Risk Score Trends by Department',
                     template='plotly_white')
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Risk Score",
            legend_title="Department",
            hovermode='x unified'
        )
        return fig

    def _create_department_risk_plot(self, df):
        """Create department risk comparison visualization"""
        dept_risk = df.groupby('department').agg({
            'risk_score': 'mean',
            'compliance_score': 'mean',
            'incident_count': 'sum'
        }).round(2)

        fig = go.Figure(data=[
            go.Bar(name='Risk Score',
                  x=dept_risk.index,
                  y=dept_risk['risk_score'],
                  marker_color='rgb(255, 99, 132)'),
            go.Bar(name='Compliance Score',
                  x=dept_risk.index,
                  y=dept_risk['compliance_score'],
                  marker_color='rgb(54, 162, 235)')
        ])

        fig.update_layout(
            title='Department Risk vs Compliance Comparison',
            barmode='group',
            template='plotly_white'
        )
        return fig

    def _setup_callbacks(self):
        """Setup dashboard callbacks"""
        @self.app.callback(
            [Output('risk-trend-plot', 'figure'),
             Output('department-comparison-plot', 'figure'),
             Output('risk-distribution-plot', 'figure'),
             Output('compliance-plot', 'figure'),
             Output('risk-cluster-plot', 'figure')],
            [Input('department-filter', 'value'),
             Input('date-range', 'start_date'),
             Input('date-range', 'end_date'),
             Input('risk-threshold', 'value')]
        )
        def update_graphs(department, start_date, end_date, risk_threshold):
            # Filter data based on selections
            filtered_df = self.df.copy()
            
            if department != 'All':
                filtered_df = filtered_df[filtered_df['department'] == department]
            
            filtered_df = filtered_df[
                (filtered_df['date'] >= start_date) &
                (filtered_df['date'] <= end_date)
            ]
            
            # Create updated visualizations
            risk_trend = self._create_risk_trend_plot(filtered_df)
            dept_comparison = self._create_department_risk_plot(filtered_df)
            
            # Risk distribution
            risk_dist = px.histogram(
                filtered_df,
                x='risk_score',
                color='department',
                title='Risk Score Distribution',
                template='plotly_white'
            )
            
            # Compliance overview
            compliance = px.scatter(
                filtered_df,
                x='risk_score',
                y='compliance_score',
                color='department',
                title='Risk vs Compliance',
                template='plotly_white'
            )
            
            # Risk clusters
            clusters = px.scatter(
                filtered_df,
                x='pca_1',
                y='pca_2',
                color='risk_cluster',
                title='Risk Clusters Analysis',
                template='plotly_white'
            )
            
            return risk_trend, dept_comparison, risk_dist, compliance, clusters
