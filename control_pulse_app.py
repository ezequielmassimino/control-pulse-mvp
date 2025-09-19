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
import time
import base64
from io import BytesIO

# Page Configuration
st.set_page_config(
    page_title="Control Pulse - Financial Control Monitoring",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
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
if 'risk_threshold' not in st.session_state:
    st.session_state.risk_threshold = 70

# Custom CSS
def load_css():
    st.markdown("""
    <style>
    .main > div {padding-top: 2rem;}
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        border: none;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
    }
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.05);
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Load demo data
@st.cache_data
def load_data():
    """Generate demo transaction data"""
    np.random.seed(42)
    
    # Normal transactions
    normal_transactions = []
    vendors = ['Acme Corp', 'TechSupply Inc', 'Office Pro', 'Facilities Co', 'Marketing Agency']
    approvers = ['jsmith', 'mjones', 'rbrown', 'lwilson', 'kdavis']
    base_date = datetime.now() - timedelta(days=30)
    
    for i in range(40):
        trans = {
            'transaction_id': f'INV-{2900 + i}',
            'vendor_name': np.random.choice(vendors),
            'amount': abs(np.random.normal(5000, 2000)),
            'approver': np.random.choice(approvers),
            'approval_date': base_date + timedelta(days=np.random.randint(0, 30)),
            'risk_score': np.random.uniform(0, 30),
            'anomaly_type': 'Normal',
            'status': 'Processed'
        }
        normal_transactions.append(trans)
    
    # Anomalies
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
            'ai_explanation': 'New vendor, first transaction 5x higher than typical'
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
            'ai_explanation': '48 approvals in 2 minutes by same user'
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
            'ai_explanation': '15 invoices just below $1000 approval threshold'
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
            'ai_explanation': 'Suspiciously round amount with no documentation'
        }
    ]
    
    all_transactions = pd.DataFrame(normal_transactions + anomalies)
    all_transactions['amount'] = all_transactions['amount'].round(2)
    return all_transactions

def export_to_excel(dataframe):
    """Export dataframe to Excel"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        dataframe.to_excel(writer, index=False, sheet_name='Data')
    output.seek(0)
    return output

def show_dashboard():
    """Main dashboard view"""
    st.title("⚡ Control Pulse")
    st.markdown("### Financial Controls in Real-Time")
    
    # User greeting and settings
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.markdown("#### Welcome back, Rachel!")
        st.markdown("You have **5 exceptions** requiring review today")
    with col2:
        if st.button("⚙️ Settings"):
            st.session_state.current_page = 'settings'
            st.rerun()
    with col3:
        if st.button("📁 Evidence"):
            st.session_state.current_page = 'evidence'
            st.rerun()
    
    # Load data
    df = load_data()
    exceptions = df[df['risk_score'] > st.session_state.risk_threshold].sort_values('risk_score', ascending=False)
    
    # Metrics
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🔴 Exceptions", len(exceptions), "High Priority", delta_color="inverse")
    with col2:
        st.metric("✅ Reviewed", len(st.session_state.reviewed_items), "+12 today")
    with col3:
        savings = exceptions['amount'].sum() * 0.02
        st.metric("💰 Saved This Week", f"${savings/1000:.0f}K", "+23%")
    with col4:
        st.metric("📊 Effectiveness", "94%", "+2%")
    
    # Exceptions section
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### 🎯 Exceptions Requiring Review")
    with col2:
        if st.button("📥 Export to Excel"):
            excel_file = export_to_excel(exceptions)
            st.download_button(
                label="Download Excel",
                data=excel_file,
                file_name=f"exceptions_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    # Filter
    exception_types = ['All'] + list(exceptions['anomaly_type'].unique())
    selected_type = st.selectbox("Filter by Exception Type:", exception_types)
    
    if selected_type != 'All':
        filtered_exceptions = exceptions[exceptions['anomaly_type'] == selected_type]
    else:
        filtered_exceptions = exceptions
    
    # Display exceptions
    for idx, row in filtered_exceptions.iterrows():
        col1, col2 = st.columns([5, 1])
        
        risk_icon = '🔴' if row['risk_score'] > 85 else '🟡'
        
        with col1:
            st.markdown(f"""
            **{risk_icon} {row['anomaly_type']} - {row['transaction_id']}**  
            Vendor: {row['vendor_name']} | Amount: ${row['amount']:,.2f}  
            Risk Score: {row['risk_score']:.0f}% | {row.get('ai_explanation', '')}
            """)
        
        with col2:
            if st.button("Review →", key=f"review_{idx}"):
                st.session_state.current_exception = idx
                st.session_state.current_page = 'review'
                st.rerun()
    
    # Charts
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Weekly Trend")
        dates = pd.date_range(end=datetime.now(), periods=7)
        trend_data = pd.DataFrame({
            'Date': dates,
            'Exceptions': np.random.randint(3, 8, 7),
            'Reviewed': np.random.randint(40, 60, 7)
        })
        fig = px.line(trend_data, x='Date', y=['Exceptions', 'Reviewed'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Risk Distribution")
        risk_data = pd.DataFrame({
            'Risk Level': ['High', 'Medium', 'Low'],
            'Count': [2, 3, 40]
        })
        fig = px.pie(risk_data, values='Count', names='Risk Level')
        st.plotly_chart(fig, use_container_width=True)

def show_review_screen():
    """Exception review screen"""
    df = load_data()
    exceptions = df[df['risk_score'] > st.session_state.risk_threshold].sort_values('risk_score', ascending=False)
    
    if len(exceptions) == 0:
        st.warning("No exceptions to review")
        if st.button("← Back"):
            st.session_state.current_page = 'dashboard'
            st.rerun()
        return
    
    if st.session_state.current_exception >= len(exceptions):
        st.session_state.current_exception = 0
    
    current = exceptions.iloc[st.session_state.current_exception]
    
    # Header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("← Back"):
            st.session_state.current_page = 'dashboard'
            st.rerun()
    with col2:
        st.markdown(f"### Reviewing Exception {st.session_state.current_exception + 1} of {len(exceptions)}")
    with col3:
        if st.button("Next →"):
            st.session_state.current_exception = (st.session_state.current_exception + 1) % len(exceptions)
            st.rerun()
    
    st.markdown("---")
    
    # Exception details
    st.markdown(f"# {current['anomaly_type']}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 Transaction Details")
        st.info(f"""
        **ID:** {current['transaction_id']}  
        **Vendor:** {current['vendor_name']}  
        **Amount:** ${current['amount']:,.2f}  
        **Date:** {current['approval_date'].strftime('%Y-%m-%d')}  
        **Approver:** {current['approver']}
        """)
    
    with col2:
        st.markdown("### 🤖 AI Analysis")
        st.success(f"""
        **Risk Score:** {current['risk_score']:.0f}%  
        **Explanation:** {current.get('ai_explanation', 'Pattern anomaly detected')}  
        **Recommendation:** {'Reject' if current['risk_score'] > 85 else 'Review closely'}
        """)
    
    # Actions
    st.markdown("---")
    st.markdown("### Take Action")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("✅ APPROVE", type="primary", use_container_width=True):
            st.session_state.reviewed_items.append(current['transaction_id'])
            st.success("Approved!")
            time.sleep(1)
            st.session_state.current_page = 'dashboard'
            st.rerun()
    
    with col2:
        if st.button("❌ REJECT", type="primary", use_container_width=True):
            st.session_state.reviewed_items.append(current['transaction_id'])
            st.error("Rejected!")
            time.sleep(1)
            st.session_state.current_page = 'dashboard'
            st.rerun()
    
    with col3:
        if st.button("📧 REQUEST INFO", type="primary", use_container_width=True):
            st.info("Information request sent!")

def show_evidence_repository():
    """Evidence repository screen"""
    st.title("📁 Evidence Repository")
    
    if st.button("← Back to Dashboard"):
        st.session_state.current_page = 'dashboard'
        st.rerun()
    
    st.markdown("---")
    
    # Stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Reviews", len(st.session_state.reviewed_items))
    with col2:
        st.metric("This Week", np.random.randint(20, 40))
    with col3:
        st.metric("Compliance Rate", "98%")
    
    st.markdown("---")
    
    # Recent actions
    st.markdown("### Recent Actions")
    
    if st.session_state.reviewed_items:
        for item in st.session_state.reviewed_items[-5:]:
            st.markdown(f"✅ Reviewed transaction **{item}** - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    else:
        st.info("No actions recorded yet")
    
    # Export button
    if st.button("📥 Export Audit Report", type="primary"):
        st.success("Audit report exported!")

def show_settings():
    """Settings screen"""
    st.title("⚙️ Settings")
    
    if st.button("← Back to Dashboard"):
        st.session_state.current_page = 'dashboard'
        st.rerun()
    
    st.markdown("---")
    
    # Settings sections
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎨 Appearance")
        dark_mode = st.checkbox("Dark Mode", value=st.session_state.dark_mode)
        if dark_mode != st.session_state.dark_mode:
            st.session_state.dark_mode = dark_mode
            st.rerun()
        
        st.markdown("### 🔔 Notifications")
        st.checkbox("Email Alerts", value=True)
        st.checkbox("Desktop Notifications", value=True)
        
    with col2:
        st.markdown("### ⚙️ Detection Settings")
        st.session_state.risk_threshold = st.slider(
            "Risk Threshold",
            min_value=50,
            max_value=95,
            value=st.session_state.risk_threshold,
            step=5
        )
        
        st.markdown("### 📊 Export Settings")
        st.selectbox("Default Format", ["Excel", "CSV", "PDF"])
    
    st.markdown("---")
    
    if st.button("💾 Save Settings", type="primary"):
        st.success("Settings saved!")

# Main app
def main():
    load_css()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚡ Control Pulse")
        st.markdown("---")
        
        if st.button("🏠 Dashboard"):
            st.session_state.current_page = 'dashboard'
        if st.button("⚙️ Settings"):
            st.session_state.current_page = 'settings'
        if st.button("📁 Evidence"):
            st.session_state.current_page = 'evidence'
        
        st.markdown("---")
        st.markdown("### Quick Stats")
        st.metric("Reviews Today", len(st.session_state.reviewed_items))
        st.metric("Threshold", f"{st.session_state.risk_threshold}%")
    
    # Main content
    if st.session_state.current_page == 'dashboard':
        show_dashboard()
    elif st.session_state.current_page == 'review':
        show_review_screen()
    elif st.session_state.current_page == 'evidence':
        show_evidence_repository()
    elif st.session_state.current_page == 'settings':
        show_settings()

if __name__ == "__main__":
    main()
