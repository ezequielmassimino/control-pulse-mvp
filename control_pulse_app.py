"""
Control Pulse - Financial Control Monitoring System
MVP Prototype for Customer Validation
Run with: streamlit run control_pulse_app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import time
import base64
from io import BytesIO

# Page Configuration
st.set_page_config(
    page_title="Control Pulse - Financial Control Monitoring",
    page_icon="⚡",  # Lightning bolt for real-time
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'dashboard'
if 'reviewed_items' not in st.session_state:
    st.session_state.reviewed_items = []
if 'actions_taken' not in st.session_state:
    st.session_state.actions_taken = []
if 'current_exception' not in st.session_state:
    st.session_state.current_exception = 0
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False
if 'notifications_enabled' not in st.session_state:
    st.session_state.notifications_enabled = True
if 'email_alerts' not in st.session_state:
    st.session_state.email_alerts = True
if 'risk_threshold' not in st.session_state:
    st.session_state.risk_threshold = 70
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = False
if 'data_source' not in st.session_state:
    st.session_state.data_source = 'demo'  # 'demo' or 'database'

def export_to_excel(dataframe, filename="export"):
    """Export dataframe to Excel file and create download link"""
    output = BytesIO()
    
    # Create Excel writer with xlsxwriter engine
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        dataframe.to_excel(writer, index=False, sheet_name='Data')
        
        # Get the xlsxwriter workbook and worksheet
        workbook = writer.book
        worksheet = writer.sheets['Data']
        
        # Add formatting
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#667eea',
            'font_color': 'white',
            'border': 1
        })
        
        # Write headers with formatting
        for col_num, value in enumerate(dataframe.columns.values):
            worksheet.write(0, col_num, value, header_format)
        
        # Auto-adjust column widths
        for i, col in enumerate(dataframe.columns):
            column_width = max(dataframe[col].astype(str).str.len().max(), len(col)) + 2
            worksheet.set_column(i, i, min(column_width, 50))
    
    output.seek(0)
    return output

def get_download_link(df, filename="exceptions_export"):
    """Generate a download link for dataframe as Excel"""
    excel_file = export_to_excel(df, filename)
    b64 = base64.b64encode(excel_file.read()).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx">📥 Download Excel Report</a>'
    return href
    """Create a professional SVG logo for Control Pulse"""
    logo_svg = """
    <svg width="40" height="40" viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg">
        <!-- Outer circle with gradient -->
        <defs>
            <linearGradient id="pulseGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style="stop-color:#667eea;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#764ba2;stop-opacity:1" />
            </linearGradient>
            <filter id="shadow">
                <feDropShadow dx="0" dy="1" stdDeviation="2" flood-opacity="0.2"/>
            </filter>
        </defs>
        
        <!-- Background circle -->
        <circle cx="20" cy="20" r="18" fill="url(#pulseGradient)" filter="url(#shadow)"/>
        
        <!-- Pulse line -->
        <path d="M 8 20 L 12 20 L 14 15 L 17 25 L 20 10 L 23 30 L 26 15 L 28 20 L 32 20" 
              stroke="white" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
        
        <!-- Center dot -->
        <circle cx="20" cy="20" r="2" fill="white"/>
    </svg>
    """
    return logo_svg

def get_css_styles():
    """Get CSS styles with dark mode support"""
    dark_mode = st.session_state.dark_mode
    
    if dark_mode:
        return """
        <style>
        /* Dark mode styles */
        .stApp {
            background-color: #1a1a2e;
            color: #eee;
        }
        
        /* Mobile-responsive adjustments */
        @media (max-width: 768px) {
            .block-container {
                padding: 1rem 0.5rem !important;
            }
            
            .stButton>button {
                font-size: 0.9rem !important;
                height: 2.5em !important;
                padding: 0.25rem 0.5rem !important;
            }
            
            .exception-card {
                font-size: 0.85rem !important;
                padding: 10px !important;
            }
            
            h1 { font-size: 1.5rem !important; }
            h2 { font-size: 1.2rem !important; }
            h3 { font-size: 1rem !important; }
            
            .stMetric { padding: 0.5rem !important; }
            
            [data-testid="column"] {
                width: 100% !important;
                flex: 100% !important;
            }
        }
        
        /* Dark mode specific */
        .main > div {padding-top: 2rem;}
        .stButton>button {
            width: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            height: 3em;
            border-radius: 10px;
            border: none;
            font-weight: 600;
            transition: all 0.3s;
            box-shadow: 0 2px 10px rgba(102, 126, 234, 0.3);
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.5);
        }
        .exception-card {
            background: #2a2a3e;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            border-left: 4px solid;
            word-wrap: break-word;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }
        .high-risk {border-left-color: #ff4757;}
        .medium-risk {border-left-color: #ffa502;}
        .low-risk {border-left-color: #32ff7e;}
        
        .logo-header {
            display: flex;
            align-items: center;
            gap: 15px;
            padding: 20px 0;
            border-bottom: 2px solid #667eea;
            margin-bottom: 20px;
        }
        
        .stMetric {
            background: #2a2a3e;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }
        
        div[data-testid="metric-container"] {
            background: #2a2a3e;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }
        </style>
        """
    else:
        return """
        <style>
        /* Light mode styles */
        @media (max-width: 768px) {
            .block-container {
                padding: 1rem 0.5rem !important;
            }
            
            .stButton>button {
                font-size: 0.9rem !important;
                height: 2.5em !important;
                padding: 0.25rem 0.5rem !important;
            }
            
            .exception-card {
                font-size: 0.85rem !important;
                padding: 10px !important;
            }
            
            h1 { font-size: 1.5rem !important; }
            h2 { font-size: 1.2rem !important; }
            h3 { font-size: 1rem !important; }
            
            .stMetric { padding: 0.5rem !important; }
            
            [data-testid="column"] {
                width: 100% !important;
                flex: 100% !important;
            }
        }
        
        /* Light mode specific */
        .main > div {padding-top: 2rem;}
        .stButton>button {
            width: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            height: 3em;
            border-radius: 10px;
            border: none;
            font-weight: 600;
            transition: all 0.3s;
            box-shadow: 0 2px 10px rgba(102, 126, 234, 0.2);
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        }
        .exception-card {
            background: white;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            border-left: 4px solid;
            word-wrap: break-word;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .high-risk {border-left-color: #ff4757;}
        .medium-risk {border-left-color: #ffa502;}
        .low-risk {border-left-color: #32ff7e;}
        
        .logo-header {
            display: flex;
            align-items: center;
            gap: 15px;
            padding: 20px 0;
            border-bottom: 2px solid #667eea;
            margin-bottom: 20px;
        }
        
        .professional-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        
        div[data-testid="metric-container"] {
            background: white;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        </style>
        """

# Apply CSS styles
st.markdown(get_css_styles(), unsafe_allow_html=True)

# Load and prepare data with database option
@st.cache_data
def load_data():
    """Load the transaction data - can connect to database or use demo data"""
    
    # Check data source preference
    if st.session_state.data_source == 'database':
        # Database connection template - uncomment and configure for production
        try:
            # Example database connection (uncomment and configure):
            # import sqlite3
            # conn = sqlite3.connect('your_database.db')
            # df = pd.read_sql_query("SELECT * FROM transactions WHERE risk_score > 70", conn)
            # conn.close()
            # return df
            
            # For MVP, fallback to demo data
            st.info("Database connection configured but using demo data for MVP")
            return load_demo_data()
        except Exception as e:
            st.warning(f"Database connection failed, using demo data: {str(e)}")
            return load_demo_data()
    else:
        return load_demo_data()

def load_demo_data():
    """Generate demo transaction data"""
    np.random.seed(42)
    
    # Generate normal transactions
    normal_transactions = []
    vendors = ['Acme Corp', 'TechSupply Inc', 'Office Pro', 'Facilities Co', 'Marketing Agency',
               'Consulting Group', 'Software Ltd', 'Hardware Supplies', 'Maintenance Co', 'Logistics Inc']
    approvers = ['jsmith', 'mjones', 'rbrown', 'lwilson', 'kdavis']
    base_date = datetime.now() - timedelta(days=30)
    
    for i in range(40):
        trans = {
            'transaction_id': f'INV-{2900 + i}',
            'vendor_name': np.random.choice(vendors),
            'amount': np.random.normal(5000, 2000),
            'approver': np.random.choice(approvers),
            'approval_date': base_date + timedelta(days=np.random.randint(0, 30)),
            'risk_score': np.random.uniform(0, 30),
            'anomaly_type': 'Normal',
            'status': 'Processed'
        }
        normal_transactions.append(trans)
    
    # Add diverse exception types
    anomalies = [
        {
            'transaction_id': 'INV-2947',
            'vendor_name': 'Acme Corp',
            'amount': 15420.00,
            'approver': 'jsmith',
            'approval_date': datetime.now() - timedelta(days=1),
            'risk_score': 94,
            'anomaly_type': 'Duplicate Invoice',
            'status': 'Pending Review',
            'duplicate_of': 'INV-2946',
            'ai_explanation': '94% match with INV-2946: Same vendor, same amount, 1 day apart'
        },
        {
            'transaction_id': 'VND-NEW-001',
            'vendor_name': 'TechPro LLC',
            'amount': 87650.00,
            'approver': 'mjones',
            'approval_date': datetime.now() - timedelta(hours=2),
            'risk_score': 78,
            'anomaly_type': 'Unusual Vendor Pattern',
            'status': 'Pending Review',
            'ai_explanation': 'New vendor, first transaction 5x higher than typical. No PO match found.'
        },
        {
            'transaction_id': 'APR-384',
            'vendor_name': 'Various',
            'amount': 125000.00,
            'approver': 'rbrown',
            'approval_date': datetime.now() - timedelta(hours=1),
            'risk_score': 82,
            'anomaly_type': 'Velocity Anomaly',
            'status': 'Pending Review',
            'ai_explanation': '48 approvals in 2 minutes by same user. Unusual velocity pattern detected.'
        },
        {
            'transaction_id': 'INV-3001',
            'vendor_name': 'Office Pro',
            'amount': 999.99,
            'approver': 'kdavis',
            'approval_date': datetime.now() - timedelta(hours=3),
            'risk_score': 88,
            'anomaly_type': 'Split Invoice Pattern',
            'status': 'Pending Review',
            'ai_explanation': '15 invoices of $999.99 each, just below $1000 approval threshold'
        },
        {
            'transaction_id': 'PAY-8821',
            'vendor_name': 'Consulting Group',
            'amount': 45000.00,
            'approver': 'lwilson',
            'approval_date': datetime.now() - timedelta(hours=5),
            'risk_score': 91,
            'anomaly_type': 'Round Amount Anomaly',
            'status': 'Pending Review',
            'ai_explanation': 'Suspiciously round amount ($45,000.00) with no supporting documentation'
        },
        {
            'transaction_id': 'WKD-991',
            'vendor_name': 'Maintenance Co',
            'amount': 12500.00,
            'approver': 'jsmith',
            'approval_date': datetime.now() - timedelta(hours=8),
            'risk_score': 76,
            'anomaly_type': 'Weekend Processing',
            'status': 'Pending Review',
            'ai_explanation': 'Transaction processed on Sunday at 2:30 AM - unusual timing'
        },
        {
            'transaction_id': 'VND-CHANGE-02',
            'vendor_name': 'TechSupply Inc',
            'amount': 8750.00,
            'approver': 'mjones',
            'approval_date': datetime.now() - timedelta(hours=12),
            'risk_score': 85,
            'anomaly_type': 'Vendor Detail Change',
            'status': 'Pending Review',
            'ai_explanation': 'Bank account changed 2 days ago, first payment to new account'
        },
        {
            'transaction_id': 'BEN-109',
            'vendor_name': 'Unknown Beneficiary',
            'amount': 5600.00,
            'approver': 'rbrown',
            'approval_date': datetime.now() - timedelta(days=2),
            'risk_score': 73,
            'anomaly_type': 'Missing Vendor Registration',
            'status': 'Pending Review',
            'ai_explanation': 'Payment to unregistered vendor, no W-9 or tax documentation on file'
        },
        {
            'transaction_id': 'SEQ-BREAK-44',
            'vendor_name': 'Office Pro',
            'amount': 3200.00,
            'approver': 'kdavis',
            'approval_date': datetime.now() - timedelta(days=1, hours=6),
            'risk_score': 79,
            'anomaly_type': 'Sequential Break',
            'status': 'Pending Review',
            'ai_explanation': 'Invoice numbers 1044, 1045, 1047 - number 1046 is missing'
        },
        {
            'transaction_id': 'GEO-ALERT-22',
            'vendor_name': 'International Supplies',
            'amount': 22000.00,
            'approver': 'lwilson',
            'approval_date': datetime.now() - timedelta(hours=15),
            'risk_score': 81,
            'anomaly_type': 'Geographic Anomaly',
            'status': 'Pending Review',
            'ai_explanation': 'Vendor location changed from USA to high-risk jurisdiction'
        }
    ]
    
    all_transactions = pd.DataFrame(normal_transactions + anomalies)
    all_transactions['amount'] = all_transactions['amount'].round(2)
    return all_transactions

def calculate_metrics(df):
    """Calculate key metrics for dashboard"""
    total_transactions = len(df)
    exceptions = df[df['risk_score'] > st.session_state.risk_threshold]
    reviewed = df[df['status'] == 'Processed']
    
    # Calculate potential savings (demo calculation)
    potential_savings = exceptions['amount'].sum() * 0.02  # Assume 2% error rate
    
    return {
        'total': total_transactions,
        'exceptions': len(exceptions),
        'reviewed': len(reviewed),
        'savings': potential_savings,
        'effectiveness': 94
    }

def is_mobile():
    """Check if user is on mobile device"""
    return st.session_state.get('mobile_view', False)

def show_dashboard():
    """Main dashboard view with mobile responsiveness"""
    # Professional header with logo
    logo_svg = create_logo_svg()
    st.markdown(f"""
    <div class="logo-header">
        {logo_svg}
        <div>
            <h1 style="margin: 0; font-size: 2rem; font-weight: 700;">Control Pulse</h1>
            <p style="margin: 0; opacity: 0.8;">Financial Controls in Real-Time</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # User greeting
    if not is_mobile():
        col1, col2, col3 = st.columns([2, 1, 1])
    else:
        col1 = st.container()
        col2 = None
        col3 = None
    
    with col1:
        st.markdown("### Good morning, Rachel!")
        st.markdown("You have **10 exceptions** requiring review")
    
    if not is_mobile() and col2 is not None:
        with col2:
            if st.button("🔔 Notifications"):
                st.info(f"{'🔔 Enabled' if st.session_state.notifications_enabled else '🔕 Disabled'}")
    
    if not is_mobile() and col3 is not None:
        with col3:
            if st.button("⚙️ Settings"):
                st.session_state.current_page = 'settings'
                st.rerun()
    
    # Load data and calculate metrics
    df = load_data()
    metrics = calculate_metrics(df)
    exceptions = df[df['risk_score'] > st.session_state.risk_threshold].sort_values('risk_score', ascending=False)
    
    # Display key metrics
    st.markdown("---")
    
    if is_mobile():
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🔴 Exceptions", metrics['exceptions'], "High Priority", delta_color="inverse")
        with col2:
            st.metric("✅ Reviewed", metrics['reviewed'], "+12")
        
        col3, col4 = st.columns(2)
        with col3:
            st.metric("💰 Saved", f"${metrics['savings']/1000:.0f}K", "+23%")
        with col4:
            st.metric("📊 Effectiveness", f"{metrics['effectiveness']}%", "+2%")
    else:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🔴 Exceptions", metrics['exceptions'], "High Priority", delta_color="inverse")
        with col2:
            st.metric("✅ Reviewed", metrics['reviewed'], "+12")
        with col3:
            st.metric("💰 Saved", f"${metrics['savings']/1000:.0f}K", "+23%")
        with col4:
            st.metric("📊 Effectiveness", f"{metrics['effectiveness']}%", "+2%")
    
    # Exception type filter with export button
    st.markdown("---")
    col1, col2 = st.columns([3, 1]) if not is_mobile() else [st.container(), st.container()]
    
    with col1:
        st.markdown("### 🎯 Exceptions Requiring Review")
        exception_types = ['All'] + list(exceptions['anomaly_type'].unique())
        selected_type = st.selectbox("Filter by Exception Type:", exception_types, index=0)
    
    with col2:
        st.markdown("### 📊 Export")
        if st.button("📥 Export to Excel", use_container_width=True):
            # Prepare export data
            export_df = exceptions[['transaction_id', 'vendor_name', 'amount', 'risk_score', 
                                   'anomaly_type', 'ai_explanation', 'approval_date', 'status']]
            st.markdown(get_download_link(export_df, "control_pulse_exceptions"), unsafe_allow_html=True)
    
    if selected_type != 'All':
        filtered_exceptions = exceptions[exceptions['anomaly_type'] == selected_type]
    else:
        filtered_exceptions = exceptions
    
    st.markdown(f"*Showing {len(filtered_exceptions)} exception(s)*")
    
    # Display exceptions
    for idx, row in filtered_exceptions.iterrows():
        risk_class = 'high-risk' if row['risk_score'] > 85 else 'medium-risk' if row['risk_score'] > 75 else 'low-risk'
        risk_icon = '🔴' if row['risk_score'] > 85 else '🟡' if row['risk_score'] > 75 else '🟠'
        
        if is_mobile():
            st.markdown(f"""
            <div class="exception-card {risk_class}">
                <strong>{risk_icon} {row['anomaly_type']}</strong><br>
                <strong>{row['transaction_id']}</strong><br>
                Vendor: {row['vendor_name']}<br>
                Amount: ${row['amount']:,.2f}<br>
                Risk Score: {row['risk_score']:.0f}%<br>
                <small>{row.get('ai_explanation', '')}</small>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(f"Review This Exception →", key=f"review_{idx}"):
                st.session_state.current_exception = list(filtered_exceptions.index).index(idx)
                st.session_state.current_page = 'review'
                st.rerun()
        else:
            col1, col2 = st.columns([5, 1])
            with col1:
                st.markdown(f"""
                <div class="exception-card {risk_class}">
                    <strong>{risk_icon} {row['anomaly_type']} | {row['transaction_id']}</strong><br>
                    Vendor: {row['vendor_name']} | Amount: ${row['amount']:,.2f}<br>
                    AI Confidence: {row['risk_score']:.0f}% | {row.get('ai_explanation', '')}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                if st.button(f"Review →", key=f"review_{idx}"):
                    st.session_state.current_exception = list(filtered_exceptions.index).index(idx)
                    st.session_state.current_page = 'review'
                    st.rerun()
    
    # Quick stats visualization
    st.markdown("---")
    
    if is_mobile():
        st.markdown("### 📈 Weekly Trend")
        dates = pd.date_range(end=datetime.now(), periods=7)
        trend_data = pd.DataFrame({
            'Date': dates,
            'Exceptions': np.random.randint(8, 15, 7),
            'Reviewed': np.random.randint(40, 60, 7)
        })
        
        fig = px.line(trend_data, x='Date', y=['Exceptions', 'Reviewed'],
                     title="Exception Detection Trend",
                     color_discrete_map={'Exceptions': '#ff4757', 'Reviewed': '#32ff7e'})
        fig.update_layout(height=250, showlegend=True, margin=dict(l=0, r=0, t=30, b=0),
                         paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 🎯 Risk Distribution")
        risk_dist = pd.DataFrame({
            'Risk Level': ['High', 'Medium', 'Low'],
            'Count': [4, 4, 42]
        })
        
        fig = px.pie(risk_dist, values='Count', names='Risk Level',
                    color_discrete_map={'High': '#ff4757', 'Medium': '#ffa502', 'Low': '#32ff7e'},
                    title="Current Risk Distribution")
        fig.update_layout(height=250, margin=dict(l=0, r=0, t=30, b=0),
                         paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Weekly Trend")
            dates = pd.date_range(end=datetime.now(), periods=7)
            trend_data = pd.DataFrame({
                'Date': dates,
                'Exceptions': np.random.randint(8, 15, 7),
                'Reviewed': np.random.randint(40, 60, 7)
            })
            
            fig = px.line(trend_data, x='Date', y=['Exceptions', 'Reviewed'],
                         title="Exception Detection Trend",
                         color_discrete_map={'Exceptions': '#ff4757', 'Reviewed': '#32ff7e'})
            fig.update_layout(height=300, showlegend=True,
                             paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🎯 Risk Distribution")
            risk_dist = pd.DataFrame({
                'Risk Level': ['High', 'Medium', 'Low'],
                'Count': [4, 4, 42]
            })
            
            fig = px.pie(risk_dist, values='Count', names='Risk Level',
                        color_discrete_map={'High': '#ff4757', 'Medium': '#ffa502', 'Low': '#32ff7e'},
                        title="Current Risk Distribution")
            fig.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

def show_settings():
    """Settings page for user preferences"""
    # Header
    logo_svg = create_logo_svg()
    st.markdown(f"""
    <div class="logo-header">
        {logo_svg}
        <div>
            <h1 style="margin: 0; font-size: 2rem; font-weight: 700;">Settings</h1>
            <p style="margin: 0; opacity: 0.8;">Configure your Control Pulse preferences</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("← Back to Dashboard"):
        st.session_state.current_page = 'dashboard'
        st.rerun()
    
    st.markdown("---")
    
    # Settings sections
    col1, col2 = st.columns(2) if not is_mobile() else [st.container(), st.container()]
    
    with col1:
        st.markdown("### 🎨 Appearance")
        dark_mode = st.checkbox("🌙 Dark Mode", value=st.session_state.dark_mode)
        if dark_mode != st.session_state.dark_mode:
            st.session_state.dark_mode = dark_mode
            st.rerun()
        
        mobile_view = st.checkbox("📱 Mobile View", value=st.session_state.get('mobile_view', False))
        if mobile_view != st.session_state.get('mobile_view', False):
            st.session_state.mobile_view = mobile_view
            st.rerun()
        
        st.markdown("### 🔔 Notifications")
        st.session_state.notifications_enabled = st.checkbox(
            "Enable Desktop Notifications",
            value=st.session_state.notifications_enabled
        )
        
        st.session_state.email_alerts = st.checkbox(
            "Enable Email Alerts",
            value=st.session_state.email_alerts
        )
        
        if st.session_state.email_alerts:
            email = st.text_input("Email Address", value="rachel.chen@company.com")
            alert_frequency = st.selectbox(
                "Alert Frequency",
                ["Immediate", "Hourly", "Daily Summary", "Weekly Report"]
            )
    
    with col2:
        st.markdown("### ⚙️ Detection Settings")
        
        st.session_state.risk_threshold = st.slider(
            "Risk Score Threshold",
            min_value=50,
            max_value=95,
            value=st.session_state.risk_threshold,
            step=5,
            help="Transactions with risk scores above this threshold will be flagged"
        )
        
        st.session_state.auto_refresh = st.checkbox(
            "Auto-refresh Dashboard",
            value=st.session_state.auto_refresh
        )
        
        if st.session_state.auto_refresh:
            refresh_interval = st.selectbox(
                "Refresh Interval",
                ["30 seconds", "1 minute", "5 minutes", "15 minutes"]
            )
        
    st.markdown("---")
    st.markdown("### 📊 Report Preferences")
    
    col1, col2 = st.columns(2) if not is_mobile() else [st.container(), st.container()]
    
    with col1:
        report_format = st.selectbox(
            "Default Export Format",
            ["Excel (.xlsx)", "CSV (.csv)", "PDF (coming soon)"]
        )
        
        include_evidence = st.checkbox("Include Evidence in Reports", value=True)
        include_ai_reasoning = st.checkbox("Include AI Reasoning", value=True)
    
    with col2:
        st.markdown("### 🔗 Data Source")
        data_source = st.radio(
            "Select Data Source:",
            ["Demo Data (MVP)", "Database Connection"],
            index=0 if st.session_state.data_source == 'demo' else 1
        )
        
        if data_source == "Database Connection":
            st.session_state.data_source = 'database'
            st.info("📌 Database Template Ready")
            with st.expander("Database Configuration"):
                st.code("""
# Example configuration (uncomment in production):
# DATABASE_CONFIG = {
#     'host': 'your-host',
#     'database': 'your-database',
#     'user': 'your-username',
#     'password': 'your-password',
#     'port': 5432
# }
                """)
                st.text_input("Host", placeholder="localhost")
                st.text_input("Database", placeholder="control_pulse_db")
                st.text_input("Username", placeholder="admin")
                st.text_input("Password", type="password", placeholder="••••••••")
                
                if st.button("Test Connection"):
                    st.warning("Using demo data for MVP. Database connection ready for production.")
        else:
            st.session_state.data_source = 'demo'
    
    st.markdown("---")
    st.markdown("### 👤 User Profile")
    
    col1, col2 = st.columns(2) if not is_mobile() else [st.container(), st.container()]
    
    with col1:
        st.text_input("Name", value="Rachel Chen")
        st.text_input("Title", value="Senior Financial Controller")
        st.text_input("Department", value="Finance")
    
    with col2:
        st.text_input("Employee ID", value="EMP-2847")
        st.selectbox("Approval Authority", ["$10,000", "$50,000", "$100,000", "$500,000", "Unlimited"])
        st.selectbox("Time Zone", ["EST", "CST", "MST", "PST", "UTC"])
    
    st.markdown("---")
    st.markdown("### 🔐 Security Settings")
    
    col1, col2 = st.columns(2) if not is_mobile() else [st.container(), st.container()]
    
    with col1:
        st.checkbox("Two-Factor Authentication", value=True)
        st.checkbox("Require Re-authentication for High-Risk Actions", value=True)
        session_timeout = st.selectbox("Session Timeout", ["15 minutes", "30 minutes", "1 hour", "4 hours"])
    
    with col2:
        st.checkbox("Audit All Actions", value=True)
        st.checkbox("Screen Recording for Evidence", value=False)
        st.selectbox("Data Retention Period", ["30 days", "90 days", "1 year", "7 years"])
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3) if not is_mobile() else [st.container(), st.container(), st.container()]
    
    with col1:
        if st.button("💾 Save Settings", type="primary", use_container_width=True):
            st.success("✅ Settings saved successfully!")
    
    with col2:
        if st.button("🔄 Reset to Defaults", use_container_width=True):
            st.session_state.dark_mode = False
            st.session_state.notifications_enabled = True
            st.session_state.email_alerts = True
            st.session_state.risk_threshold = 70
            st.session_state.auto_refresh = False
            st.info("Settings reset to defaults")
            st.rerun()
    
    with col3:
        if st.button("📤 Export Settings", use_container_width=True):
            st.info("Settings exported to settings_config.json")
