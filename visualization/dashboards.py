import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, html, dcc, Input, Output, State, callback
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy import stats


class DashboardCreator:
    def __init__(self):
        self.app = Dash(__name__)
        self.theme = {
            'background': '#F8F9FA',
            'card_bg': '#FFFFFF',
            'primary': '#3B82F6',
            'secondary': '#10B981',
            'warning': '#F59E0B',
            'danger': '#EF4444',
            'text': '#1F2937',
            'text_secondary': '#6B7280',
            'border': '#E5E7EB'
        }
        self.df = None
        self.risk_metrics = None

    def filter_data(self, department, start_date, end_date):
        """Filter dataframe based on department and date range"""
        filtered_df = self.df.copy()
        
        # Convert dates if they're strings
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Apply department filter
        if department != 'All':
            filtered_df = filtered_df[filtered_df['department'] == department]
            
        # Apply date filter
        filtered_df = filtered_df[
            (filtered_df['date'] >= start_date) & 
            (filtered_df['date'] <= end_date)
        ]
        
        if len(filtered_df) == 0:
            raise ValueError("No data available for selected filters")
            
        return filtered_df

    def create_risk_dashboard(self, df, risk_metrics):
        """Create comprehensive risk management dashboard"""
        self.df = df
        self.risk_metrics = risk_metrics
        
        # Validate input data
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame")
        
        required_columns = ['date', 'department', 'risk_score', 'compliance_score', 
                          'incident_count', 'recovery_time_hours']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Initialize callbacks before layout
        self.setup_callbacks()

        # Create the layout
        self.app.layout = html.Div([
            # Navigation Bar
            self.create_navigation_bar(),
            
            # Main Content Container
            html.Div([
                # Left Sidebar
                self.create_sidebar(),
                
                # Main Content Area
                html.Div([
                    # Header Section
                    self.create_header_section(),
                    
                    # Filters Section
                    self.create_filters_section(),
                    
                    # Main Grid
                    html.Div([
                        # Left Column - Trends and Analysis
                        html.Div([
                            html.Div([
                                self.create_risk_trends_section()
                            ], className="mb-4"),
                            html.Div([
                                self.create_risk_analysis_section()
                            ])
                        ], className="w-2/3 pr-4"),
                        
                        # Right Column - Alerts and Recommendations
                        html.Div([
                            html.Div([
                                self.create_alerts_section()
                            ], className="mb-4"),
                            html.Div([
                                self.create_recommendations_section()
                            ])
                        ], className="w-1/3")
                    ], className="flex mb-4"),
                    
                    # Bottom Section - Detailed Analytics
                    html.Div([
                        self.create_detailed_analytics_tabs()
                    ], className="w-full")
                    
                ], className="flex-1 p-6 bg-gray-100 overflow-auto")
            ], className="flex min-h-screen bg-gray-100")
        ], className="min-h-screen bg-gray-50")

        return self.app

    def setup_callbacks(self):
        @self.app.callback(
            [Output('overall-risk-score', 'children'),
             Output('risk-trend', 'children'),
             Output('risk-details', 'children'),
             Output('compliance-rate', 'children'),
             Output('compliance-trend', 'children'),
             Output('compliance-details', 'children'),
             Output('incident-count', 'children'),
             Output('incident-trend', 'children'),
             Output('incident-details', 'children'),
             Output('recovery-time', 'children'),
             Output('recovery-trend', 'children'),
             Output('recovery-details', 'children'),
             Output('risk-trends-chart', 'figure'),
             Output('risk-heatmap', 'figure'),
             Output('dept-comparison-chart', 'figure'),
             Output('risk-distribution-chart', 'figure'),
             Output('trend-analysis-chart', 'figure'),
             Output('risk-clusters-chart', 'figure'),
             Output('time-series-chart', 'figure'),
             Output('correlation-chart', 'figure'),
             Output('alerts-content', 'children'),
             Output('recommendations-content', 'children'),
             Output('sidebar-alerts', 'children')],
            [Input('department-filter', 'value'),
             Input('date-range', 'start_date'),
             Input('date-range', 'end_date'),
             Input('analysis-options', 'value')],
            prevent_initial_call=False
        )
        def update_dashboard(department, start_date, end_date, analysis_options):
            if not all([department, start_date, end_date]):
                department = 'All'
                start_date = self.df['date'].min()
                end_date = self.df['date'].max()
                analysis_options = ['trends']

            try:
                # Filter data
                filtered_df = self.filter_data(department, start_date, end_date)
                
                # Calculate base metrics
                metrics = self.calculate_metrics(filtered_df)
                
                # Format metrics for display
                formatted_metrics = self.format_metrics(metrics)
                
                # Generate figures with error handling
                try:
                    figures = self.generate_figures(filtered_df, analysis_options or ['trends'])
                except Exception as e:
                    print(f"Error generating figures: {str(e)}")
                    figures = self.generate_empty_figures()
                
                # Generate content
                try:
                    content = self.generate_content(metrics, filtered_df)
                except Exception as e:
                    print(f"Error generating content: {str(e)}")
                    content = self.generate_empty_content()
                
                return (*formatted_metrics, *figures.values(), *content.values())
                
            except Exception as e:
                print(f"Error in dashboard update: {str(e)}")
                return self.generate_error_state()

    def calculate_metrics(self, df):
        try:
            return {
                'risk_score': df['risk_score'].mean(),
                'risk_trend': self.calculate_trend(df.groupby(df['date'].dt.date)['risk_score'].mean()),
                'compliance_rate': df['compliance_score'].mean(),
                'compliance_trend': self.calculate_trend(df.groupby(df['date'].dt.date)['compliance_score'].mean()),
                'incident_count': len(df[df['risk_score'] > 80]),
                'incident_trend': self.calculate_trend(df.groupby(df['date'].dt.date)['incident_count'].sum()),
                'recovery_time': df['recovery_time_hours'].mean(),
                'recovery_trend': self.calculate_trend(df.groupby(df['date'].dt.date)['recovery_time_hours'].mean())
            }
        except Exception as e:
            print(f"Error calculating metrics: {str(e)}")
            return self.generate_empty_metrics()

    def calculate_trend(self, series):
        try:
            if len(series) < 2:
                return 0
            # Use first and last non-NaN values
            first_valid = series.dropna().iloc[0]
            last_valid = series.dropna().iloc[-1]
            return ((last_valid - first_valid) / first_valid * 100) if first_valid != 0 else 0
        except Exception as e:
            print(f"Error calculating trend: {str(e)}")
            return 0

    def generate_error_state(self):
        """Generate safe default values for all callback outputs"""
        empty_metrics = ['N/A'] * 12  # For all metric outputs
        empty_figures = [go.Figure()] * 8  # For all chart outputs
        empty_content = [html.Div("No data available")] * 3  # For alerts and recommendations
        return [*empty_metrics, *empty_figures, *empty_content]

    def generate_empty_metrics(self):
        """Generate empty metrics structure"""
        return {
            'risk_score': 0,
            'risk_trend': 0,
            'compliance_rate': 0,
            'compliance_trend': 0,
            'incident_count': 0,
            'incident_trend': 0,
            'recovery_time': 0,
            'recovery_trend': 0
        }



    def generate_empty_figures(self):
        """Generate empty figures for all charts"""
        empty_fig = go.Figure()
        empty_fig.update_layout(
            annotations=[{
                'text': 'No data available',
                'xref': 'paper',
                'yref': 'paper',
                'showarrow': False,
                'font': {'size': 20}
            }]
        )
        
        return {
            'risk_trends': empty_fig,
            'risk_heatmap': empty_fig,
            'dept_comparison': empty_fig,
            'risk_distribution': empty_fig,
            'trend_analysis': empty_fig,
            'risk_clusters': empty_fig,
            'time_series': empty_fig,
            'correlation': empty_fig
        }

    def generate_empty_content(self):
        """Generate empty content structure"""
        return {
            'alerts': [],
            'recommendations': [],
            'sidebar_alerts': []
        }

    def create_navigation_bar(self):
        return html.Nav([
            html.Div([
                html.H1("Enterprise Risk Management", className="text-xl font-bold text-white"),
                html.Div([
                    html.Button("Refresh Data", id="refresh-data", className="bg-blue-600 px-4 py-2 rounded"),
                    html.Button("Export Report", id="export-report", className="bg-green-600 px-4 py-2 rounded ml-2"),
                    html.Button("Settings", id="settings-btn", className="bg-gray-600 px-4 py-2 rounded ml-2"),
                ])
            ], className="container mx-auto flex justify-between items-center")
        ], className="bg-gray-800 text-white p-4")



    def create_sidebar(self):
        return html.Div([
            html.Div([
                html.H2("Quick Navigation", className="text-lg font-bold mb-4"),
                html.Ul([
                    html.Li("Risk Overview", className="mb-2 hover:bg-gray-200 p-2 rounded cursor-pointer"),
                    html.Li("Department Analysis", className="mb-2 hover:bg-gray-200 p-2 rounded cursor-pointer"),
                    html.Li("Compliance Metrics", className="mb-2 hover:bg-gray-200 p-2 rounded cursor-pointer"),
                    html.Li("Historical Trends", className="mb-2 hover:bg-gray-200 p-2 rounded cursor-pointer"),
                ])
            ], className="mb-8"),
            
            html.Div([
                html.H2("Alerts", className="text-lg font-bold mb-4"),
                html.Div(id='sidebar-alerts', className="space-y-2")
            ])
        ], className="w-64 bg-white p-4 shadow-lg")

    def create_header_section(self):
        return html.Div([
            html.Div([
                html.H1("Risk Analytics Dashboard", className="text-2xl font-bold"),
                html.Div([
                    html.Span("Last Updated: ", className="text-gray-500"),
                    html.Span(datetime.now().strftime('%Y-%m-%d %H:%M'), 
                             className="font-semibold")
                ], className="text-sm")
            ], className="flex justify-between items-baseline mb-4"),
            
            # Key Metrics Grid
            html.Div([
                self.create_metric_card(
                    "Overall Risk Score",
                    "overall-risk-score",
                    "risk-trend",
                    "risk-details"
                ),
                self.create_metric_card(
                    "Compliance Rate",
                    "compliance-rate",
                    "compliance-trend",
                    "compliance-details"
                ),
                self.create_metric_card(
                    "High Risk Incidents",
                    "incident-count",
                    "incident-trend",
                    "incident-details"
                ),
                self.create_metric_card(
                    "Recovery Time",
                    "recovery-time",
                    "recovery-trend",
                    "recovery-details"
                )
            ], className="grid grid-cols-4 gap-4")
        ], className="mb-6")

    def create_metric_card(self, title, value_id, trend_id, detail_id):
        return html.Div([
            html.H3(title, className="text-lg font-medium text-gray-700"),
            html.Div([
                html.Div(id=value_id, className="text-3xl font-bold"),
                html.Div(id=trend_id, className="ml-2")
            ], className="flex items-center"),
            html.Div(id=detail_id, className="text-sm text-gray-500 mt-1")
        ], className="bg-white p-4 rounded-lg shadow")

    def create_filters_section(self):
        return html.Div([
            html.Div([
                html.Label("Department", className="block text-sm font-medium text-gray-700"),
                dcc.Dropdown(
                    id='department-filter',
                    options=[{'label': 'All Departments', 'value': 'All'}] +
                            [{'label': d, 'value': d} for d in self.df['department'].unique()],
                    value='All',
                    className="mt-1"
                )
            ], className="w-1/4"),
            
            html.Div([
                html.Label("Time Period", className="block text-sm font-medium text-gray-700"),
                dcc.DatePickerRange(
                    id='date-range',
                    start_date=self.df['date'].min(),
                    end_date=self.df['date'].max(),
                    className="mt-1"
                )
            ], className="w-2/4"),
            
            html.Div([
                html.Label("Analysis Options", className="block text-sm font-medium text-gray-700"),
                dcc.Checklist(
                    id='analysis-options',
                    options=[
                        {'label': ' Show Trends', 'value': 'trends'},
                        {'label': ' Show Forecasts', 'value': 'forecasts'},
                        {'label': ' Show Anomalies', 'value': 'anomalies'}
                    ],
                    value=['trends'],
                    className="mt-2 space-y-1"
                )
            ], className="w-1/4")
        ], className="flex justify-between items-start bg-white p-4 rounded-lg shadow mb-6")

    def create_risk_trends_section(self):
        return html.Div([
            html.Div([
                html.H2("Risk Score Trends", className="text-xl font-bold"),
                html.Button(
                    "⋮",
                    id="trend-options-btn",
                    className="text-gray-500 hover:text-gray-700"
                )
            ], className="flex justify-between items-center mb-4"),
            
            dcc.Graph(id='risk-trends-chart'),
            
            html.Div([
                dcc.Graph(id='risk-heatmap')
            ], className="mt-4")
        ], className="bg-white p-4 rounded-lg shadow")

    def create_risk_analysis_section(self):
        return html.Div([
            html.H2("Risk Analysis", className="text-xl font-bold mb-4"),
            dcc.Tabs([
                dcc.Tab(label='Department Comparison', children=[
                    dcc.Graph(id='dept-comparison-chart')
                ]),
                dcc.Tab(label='Risk Distribution', children=[
                    dcc.Graph(id='risk-distribution-chart')
                ]),
                dcc.Tab(label='Trend Analysis', children=[
                    dcc.Graph(id='trend-analysis-chart')
                ])
            ])
        ], className="bg-white p-4 rounded-lg shadow mt-4")

    def create_alerts_section(self):
        return html.Div([
            html.H2("Risk Alerts", className="text-xl font-bold mb-4"),
            html.Div(id='alerts-content', className="space-y-2")
        ], className="bg-white p-4 rounded-lg shadow")

    def create_recommendations_section(self):
        return html.Div([
            html.H2("Recommendations", className="text-xl font-bold mb-4"),
            html.Div(id='recommendations-content', className="space-y-2")
        ], className="bg-white p-4 rounded-lg shadow mt-4")

    def create_detailed_analytics_tabs(self):
        return html.Div([
            html.H2("Detailed Analytics", className="text-xl font-bold mb-4"),
            dcc.Tabs([
                dcc.Tab(label='Risk Clusters', children=[
                    dcc.Graph(id='risk-clusters-chart')
                ]),
                dcc.Tab(label='Time Series Analysis', children=[
                    dcc.Graph(id='time-series-chart')
                ]),
                dcc.Tab(label='Correlations', children=[
                    dcc.Graph(id='correlation-chart')
                ])
            ])
        ], className="bg-white p-4 rounded-lg shadow mt-4")


        def update_dashboard(self, department_value, start_date, end_date, analysis_options):
            # Set default values
            department = department_value if department_value is not None else 'All'
            start_date = start_date if start_date is not None else self.df['date'].min()
            end_date = end_date if end_date is not None else self.df['date'].max()
            analysis_options = analysis_options if analysis_options is not None else ['trends']

            try:
                # Filter data
                filtered_df = self.filter_data(department, start_date, end_date)
                
                # Calculate metrics
                metrics = self.calculate_metrics(filtered_df)
                
                # Format metrics for display
                formatted_metrics = self.format_metrics(metrics)
                
                # Generate figures
                try:
                    figures = self.generate_figures(filtered_df, analysis_options)
                except Exception as e:
                    print(f"Error generating figures: {str(e)}")
                    figures = self.generate_empty_figures()
                
                # Generate content
                try:
                    content = self.generate_content(metrics, filtered_df)
                except Exception as e:
                    print(f"Error generating content: {str(e)}")
                    content = {
                        'alerts': [],
                        'recommendations': [],
                        'sidebar_alerts': []
                    }
                
                return (*formatted_metrics, *figures.values(), *content.values())
                
            except Exception as e:
                print(f"Error in dashboard update: {str(e)}")
                return self.generate_error_state()

    def calculate_trend(self, series):
        if len(series) < 2:
            return 0
        return (series.iloc[-1] - series.iloc[0]) / series.iloc[0] * 100

    def format_metrics(self, metrics):
        """Format all metrics for display"""
        def format_trend(value):
            if value > 0:
                return html.Span("↑", className="text-red-500")
            elif value < 0:
                return html.Span("↓", className="text-green-500")
            return html.Span("→", className="text-gray-500")
            
        return [
            f"{metrics['risk_score']:.1f}",
            format_trend(metrics['risk_trend']),
            f"Trend: {metrics['risk_trend']:.1f}%",
            
            f"{metrics['compliance_rate']:.1f}%",
            format_trend(metrics['compliance_trend']),
            f"Trend: {metrics['compliance_trend']:.1f}%",
            
            str(metrics['incident_count']),
            format_trend(metrics['incident_trend']),
            f"Trend: {metrics['incident_trend']:.1f}%",
            
            f"{metrics['recovery_time']:.1f}h",
            format_trend(metrics['recovery_trend']),
            f"Trend: {metrics['recovery_trend']:.1f}%"
        ]

    def generate_figures(self, df, analysis_options):
        return {
            'risk_trends': self.create_risk_trends_plot(df, analysis_options),
            'risk_heatmap': self.create_risk_heatmap(df),
            'dept_comparison': self.create_department_comparison(df),
            'risk_distribution': self.create_risk_distribution(df),
            'trend_analysis': self.create_trend_analysis(df),
            'risk_clusters': self.create_risk_clusters(df),
            'time_series': self.create_time_series(df, analysis_options),
            'correlation': self.create_correlation_analysis(df)
        }

    def create_risk_trends_plot(self, df, analysis_options):
        fig = go.Figure()
        
        # Add actual risk trends
        for dept in df['department'].unique():
            dept_data = df[df['department'] == dept]
            fig.add_trace(go.Scatter(
                x=dept_data['date'],
                y=dept_data['risk_score'],
                name=f"{dept} Risk",
                mode='lines',
                line=dict(width=2)
            ))
            
        # Add forecasts if enabled
        if 'forecasts' in analysis_options:
            for dept in df['department'].unique():
                dept_data = df[df['department'] == dept]
                forecast = self.calculate_forecast(dept_data['risk_score'])
                fig.add_trace(go.Scatter(
                    x=dept_data['date'].iloc[-len(forecast):],
                    y=forecast,
                    name=f"{dept} Forecast",
                    line=dict(dash='dash'),
                    mode='lines'
                ))
                
        # Add anomalies if enabled
        if 'anomalies' in analysis_options:
            anomalies = self.detect_anomalies(df)
            fig.add_trace(go.Scatter(
                x=anomalies['date'],
                y=anomalies['risk_score'],
                mode='markers',
                name='Anomalies',
                marker=dict(
                    size=10,
                    color='red',
                    symbol='x'
                )
            ))
            
        fig.update_layout(
            title='Risk Score Trends by Department',
            xaxis_title='Date',
            yaxis_title='Risk Score',
            hovermode='x unified',
            showlegend=True,
            template='plotly_white'
        )
        
        return fig

    def create_risk_heatmap(self, df):
        # Create pivot table for heatmap
        pivot = df.pivot_table(
            values='risk_score',
            index='department',
            columns=pd.Grouper(key='date', freq='W'),
            aggfunc='mean'
        ).round(2)
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale='RdYlGn_r',
            text=pivot.values.round(1),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title='Risk Score')
        ))
        
        fig.update_layout(
            title='Weekly Risk Score Heatmap',
            xaxis_title='Week',
            yaxis_title='Department',
            template='plotly_white'
        )
        
        return fig

    def create_department_comparison(self, df):
        dept_metrics = df.groupby('department').agg({
            'risk_score': ['mean', 'std', 'max'],
            'compliance_score': 'mean',
            'incident_count': 'sum'
        }).round(2)

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Average Risk Score',
                'Risk Score Volatility',
                'Compliance Rate',
                'Incident Count'
            )
        )

        # Average Risk Score
        fig.add_trace(
            go.Bar(
                x=dept_metrics.index,
                y=dept_metrics['risk_score']['mean'],
                name='Avg Risk',
                marker_color='#FF4136'
            ),
            row=1, col=1
        )

        # Risk Score Volatility
        fig.add_trace(
            go.Bar(
                x=dept_metrics.index,
                y=dept_metrics['risk_score']['std'],
                name='Volatility',
                marker_color='#FF851B'
            ),
            row=1, col=2
        )

        # Compliance Rate
        fig.add_trace(
            go.Bar(
                x=dept_metrics.index,
                y=dept_metrics['compliance_score']['mean'],
                name='Compliance',
                marker_color='#2ECC40'
            ),
            row=2, col=1
        )

        # Incident Count
        fig.add_trace(
            go.Bar(
                x=dept_metrics.index,
                y=dept_metrics['incident_count']['sum'],
                name='Incidents',
                marker_color='#B10DC9'
            ),
            row=2, col=2
        )

        fig.update_layout(
            height=800,
            showlegend=False,
            title_text="Department Performance Overview",
            template='plotly_white'
        )

        return fig

    def create_risk_distribution(self, df):
        fig = make_subplots(rows=1, cols=2, subplot_titles=(
            'Risk Score Distribution',
            'Risk Score vs Compliance'
        ))

        # Risk Distribution
        for dept in df['department'].unique():
            dept_data = df[df['department'] == dept]
            fig.add_trace(
                go.Histogram(
                    x=dept_data['risk_score'],
                    name=dept,
                    opacity=0.7,
                    nbinsx=30
                ),
                row=1, col=1
            )

        # Risk vs Compliance Scatter
        fig.add_trace(
            go.Scatter(
                x=df['risk_score'],
                y=df['compliance_score'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=df['incident_count'],
                    colorscale='Viridis',
                    showscale=True
                ),
                text=df['department'],
                name='Risk vs Compliance'
            ),
            row=1, col=2
        )

        fig.update_layout(
            height=400,
            template='plotly_white',
            barmode='overlay'
        )

        return fig





    def create_trend_analysis(self, df):
        """Create trend analysis visualization"""
        fig = go.Figure()
        
        # Calculate rolling averages for different windows
        windows = [7, 14, 30]
        
        for window in windows:
            for dept in df['department'].unique():
                dept_data = df[df['department'] == dept]
                rolling_avg = dept_data['risk_score'].rolling(window=window).mean()
                
                fig.add_trace(go.Scatter(
                    x=dept_data['date'],
                    y=rolling_avg,
                    name=f"{dept} ({window}-day avg)",
                    mode='lines',
                    line=dict(width=2)
                ))
        
        # Add trend lines
        for dept in df['department'].unique():
            dept_data = df[df['department'] == dept]
            
            # Calculate linear trend
            x = (dept_data['date'] - dept_data['date'].min()).dt.days
            y = dept_data['risk_score']
            
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            
            fig.add_trace(go.Scatter(
                x=dept_data['date'],
                y=p(x),
                name=f"{dept} Trend",
                line=dict(dash='dash'),
                mode='lines'
            ))
        
        fig.update_layout(
            title='Risk Score Trend Analysis',
            xaxis_title='Date',
            yaxis_title='Risk Score',
            hovermode='x unified',
            showlegend=True,
            template='plotly_white',
            height=500
        )
        
        return fig



    def create_risk_clusters(self, df):
        # Prepare data for clustering
        features = ['risk_score', 'compliance_score', 'incident_count', 'recovery_time_hours']
        X = df[features].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=4, random_state=42)
        df['cluster'] = kmeans.fit_predict(X_scaled)
        
        # Create 3D scatter plot
        fig = go.Figure(data=[
            go.Scatter3d(
                x=df['risk_score'],
                y=df['compliance_score'],
                z=df['incident_count'],
                mode='markers',
                marker=dict(
                    size=6,
                    color=df['cluster'],
                    colorscale='Viridis',
                ),
                text=df['department'],
                hovertemplate=
                    "<b>Department:</b> %{text}<br>" +
                    "<b>Risk Score:</b> %{x:.1f}<br>" +
                    "<b>Compliance:</b> %{y:.1f}<br>" +
                    "<b>Incidents:</b> %{z}<br>"
            )
        ])
        
        fig.update_layout(
            title='Risk Clusters Analysis',
            scene=dict(
                xaxis_title='Risk Score',
                yaxis_title='Compliance Score',
                zaxis_title='Incident Count'
            ),
            template='plotly_white'
        )
        
        return fig

    def create_time_series(self, df, analysis_options):
        fig = make_subplots(rows=2, cols=1, subplot_titles=(
            'Risk Score Time Series',
            'Decomposed Trend'
        ))

        # Add time series
        for dept in df['department'].unique():
            dept_data = df[df['department'] == dept]
            fig.add_trace(
                go.Scatter(
                    x=dept_data['date'],
                    y=dept_data['risk_score'],
                    name=f"{dept} Risk",
                    mode='lines'
                ),
                row=1, col=1
            )

        # Add trend decomposition if enabled
        if 'trends' in analysis_options:
            for dept in df['department'].unique():
                dept_data = df[df['department'] == dept]
                trend = self.calculate_trend_decomposition(dept_data['risk_score'])
                fig.add_trace(
                    go.Scatter(
                        x=dept_data['date'],
                        y=trend,
                        name=f"{dept} Trend",
                        line=dict(dash='dash')
                    ),
                    row=2, col=1
                )

        fig.update_layout(
            height=800,
            template='plotly_white',
            showlegend=True
        )

        return fig

    def create_correlation_analysis(self, df):
        # Calculate correlation matrix
        corr_matrix = df[[
            'risk_score', 
            'compliance_score', 
            'incident_count', 
            'recovery_time_hours'
        ]].corr()

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            text=corr_matrix.values.round(2),
            texttemplate='%{text}',
            colorscale='RdBu',
            zmin=-1,
            zmax=1
        ))

        fig.update_layout(
            title='Correlation Analysis',
            template='plotly_white'
        )

        return fig

    def calculate_forecast(self, series, periods=10):
        # Simple exponential smoothing forecast
        alpha = 0.3
        forecast = []
        last_value = series.iloc[-1]
        
        for _ in range(periods):
            forecast.append(last_value)
            last_value = alpha * series.iloc[-1] + (1 - alpha) * last_value
            
        return forecast

    def detect_anomalies(self, df):
        # Z-score based anomaly detection
        z_scores = np.abs(stats.zscore(df['risk_score']))
        return df[z_scores > 3]  # Points more than 3 standard deviations away

    def calculate_trend_decomposition(self, series):
        # Simple moving average trend
        return series.rolling(window=7).mean()

    def generate_content(self, metrics, df):
        # Generate alerts
        alerts = self.generate_alerts(metrics, df)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(metrics, df)
        
        # Generate sidebar alerts
        sidebar_alerts = self.generate_sidebar_alerts(metrics, df)
        
        return {
            'alerts': alerts,
            'recommendations': recommendations,
            'sidebar_alerts': sidebar_alerts
        }

    def generate_alerts(self, metrics, df):
        alerts = []
        
        # High risk score alert
        if metrics['risk_score'] > 80:
            alerts.append(html.Div([
                html.Strong("Critical Risk Level ", className="text-red-600"),
                html.Span(f"Overall risk score: {metrics['risk_score']:.1f}")
            ], className="p-2 bg-red-100 rounded"))
            
        # Compliance alert
        if metrics['compliance_rate'] < 90:
            alerts.append(html.Div([
                html.Strong("Low Compliance ", className="text-yellow-600"),
                html.Span(f"Current rate: {metrics['compliance_rate']:.1f}%")
            ], className="p-2 bg-yellow-100 rounded"))
            
        return alerts

    def generate_recommendations(self, metrics, df):
        recs = []
        
        # Risk-based recommendations
        if metrics['risk_score'] > 70:
            recs.append(html.Div([
                html.Strong("Risk Mitigation Required"),
                html.P("Implement immediate risk control measures")
            ], className="p-2 border-l-4 border-red-500"))
            
        # Compliance recommendations
        if metrics['compliance_rate'] < 90:
            recs.append(html.Div([
                html.Strong("Improve Compliance"),
                html.P("Review and enhance compliance procedures")
            ], className="p-2 border-l-4 border-yellow-500"))
            
        return recs

    def generate_sidebar_alerts(self, metrics, df):
        alerts = []
        
        # Critical incidents alert
        high_risk = len(df[df['risk_score'] > 90])
        if high_risk > 0:
            alerts.append(html.Div([
                html.Strong(f"{high_risk} Critical Incidents"),
                html.P("Require immediate attention")
            ], className="p-2 bg-red-100 rounded"))
            
        return alerts

    def run_server(self, debug=True, port=8050):
        self.app.run_server(debug=debug, host='0.0.0.0', port=port)
