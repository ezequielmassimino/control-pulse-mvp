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

# Page Configuration
st.set_page_config(
    page_title="Control Pulse - Financial Control Monitoring",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main > div {padding-top: 2rem;}
.stButton>button {
    width: 100%;
    background-color: #667eea;
    color: white;
    height: 3em;
    border-radius: 10px;
    border: none;
    font-weight: bold;
    transition: all 0.3s;
}
.stButton>button:hover {
    background-color: #764ba2;
    transform: translateY(-2px);
    box-shadow: 0 5px 10px rgba(0,0,0,0.2);
}
.metric-card {
    background: white;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    text-align: center;
}
.exception-card {
    background: white;
    padding: 15px;
    border-radius: 10px;
    margin: 10px 0;
    border-left: 4px solid;
}
.high-risk {border-left-color: #ff4444;}
.medium-risk {border-left-color: #ffaa00;}
.low-risk {border-left-color: #00c851;}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'dashboard'
if 'reviewed_items' not in st.session_state:
    st.session_state.reviewed_items = []
if 'actions_taken' not in st.session_state:
    st.session_state.actions_taken = []
if 'current_exception' not in st.session_state:
    st.session_state.current_exception = 0

# Load and prepare data
@st.cache_data
def load_data():
    """Load the transaction data - in production this would connect to database/ERP"""
    # For demo, we'll generate sample data
    np.random.seed(42)
    
    # Generate normal transactions
    normal_transactions = []
    vendors = ['Acme Corp', 'TechSupply Inc', 'Office Pro', 'Facilities Co', 'Marketing Agency',
               'Consulting Group', 'Software Ltd', 'Hardware Supplies', 'Maintenance Co', 'Logistics Inc']
    approvers = ['jsmith', 'mjones', 'rbrown', 'lwilson', 'kdavis']
    base_date = datetime.now() - timedelta(days=30)
    
    for i in range(47):
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
    
    # Add specific anomalies
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
        }
    ]
    
    all_transactions = pd.DataFrame(normal_transactions + anomalies)
    all_transactions['amount'] = all_transactions['amount'].round(2)
    return all_transactions

def calculate_metrics(df):
    """Calculate key metrics for dashboard"""
    total_transactions = len(df)
    exceptions = df[df['risk_score'] > 70]
    reviewed = df[df['status'] == 'Processed']
    
    # Calculate potential savings (demo calculation)
    potential_savings = exceptions['amount'].sum() * 0.02  # Assume 2% error rate
    
    return {
        'total': total_transactions,
        'exceptions': len(exceptions),
        'reviewed': len(reviewed),
        'savings': potential_savings,
        'effectiveness': 94  # Demo metric
    }

def show_dashboard():
    """Main dashboard view"""
    st.title("🎯 Control Pulse - Financial Control Monitoring")
    
    # User greeting
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown("### Good morning, Rachel! You have **3 exceptions** requiring review")
    with col3:
        if st.button("👤 Rachel Chen | Settings"):
            st.info("Settings page (coming soon)")
    
    # Load data and calculate metrics
    df = load_data()
    metrics = calculate_metrics(df)
    exceptions = df[df['risk_score'] > 70].sort_values('risk_score', ascending=False)
    
    # Display key metrics
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🔴 Exceptions",
            value=metrics['exceptions'],
            delta="High Priority",
            delta_color="inverse"
        )
    
    with col2:
        st.metric(
            label="✅ Reviewed Today",
            value=metrics['reviewed'],
            delta="+12 vs yesterday"
        )
    
    with col3:
        st.metric(
            label="💰 Saved This Week",
            value=f"${metrics['savings']:,.0f}",
            delta="+23% vs last week"
        )
    
    with col4:
        st.metric(
            label="📊 Control Effectiveness",
            value=f"{metrics['effectiveness']}%",
            delta="+2%"
        )
    
    # Exceptions requiring review
    st.markdown("---")
    st.markdown("### 🎯 Exceptions Requiring Review")
    
    for idx, row in exceptions.iterrows():
        risk_class = 'high-risk' if row['risk_score'] > 85 else 'medium-risk'
        risk_icon = '🔴' if row['risk_score'] > 85 else '🟡'
        
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
                st.session_state.current_exception = idx
                st.session_state.current_page = 'review'
                st.rerun()
    
    # Quick stats visualization
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Weekly Trend")
        dates = pd.date_range(end=datetime.now(), periods=7)
        trend_data = pd.DataFrame({
            'Date': dates,
            'Exceptions': np.random.randint(2, 8, 7),
            'Reviewed': np.random.randint(40, 60, 7)
        })
        
        fig = px.line(trend_data, x='Date', y=['Exceptions', 'Reviewed'],
                     title="Exception Detection Trend",
                     color_discrete_map={'Exceptions': '#ff4444', 'Reviewed': '#00c851'})
        fig.update_layout(height=300, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Risk Distribution")
        risk_dist = pd.DataFrame({
            'Risk Level': ['High', 'Medium', 'Low'],
            'Count': [1, 2, 44]
        })
        
        fig = px.pie(risk_dist, values='Count', names='Risk Level',
                    color_discrete_map={'High': '#ff4444', 'Medium': '#ffaa00', 'Low': '#00c851'},
                    title="Current Risk Distribution")
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

def show_review_screen():
    """Detailed review screen for exceptions"""
    df = load_data()
    exceptions = df[df['risk_score'] > 70].sort_values('risk_score', ascending=False)
    
    if st.session_state.current_exception >= len(exceptions):
        st.session_state.current_exception = 0
    
    current = exceptions.iloc[st.session_state.current_exception]
    
    # Header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("← Back to Dashboard"):
            st.session_state.current_page = 'dashboard'
            st.rerun()
    
    with col2:
        st.markdown(f"### Exception {st.session_state.current_exception + 1} of {len(exceptions)}")
    
    with col3:
        if st.button("Next →"):
            st.session_state.current_exception = (st.session_state.current_exception + 1) % len(exceptions)
            st.rerun()
    
    st.markdown("---")
    
    # Exception details
    risk_icon = '🔴' if current['risk_score'] > 85 else '🟡'
    st.markdown(f"# {risk_icon} {current['anomaly_type']} Detected")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 Transaction Details")
        st.info(f"""
        **Transaction ID:** {current['transaction_id']}
        **Vendor:** {current['vendor_name']}
        **Amount:** ${current['amount']:,.2f}
        **Approver:** {current['approver']}
        **Date:** {current['approval_date'].strftime('%Y-%m-%d %H:%M')}
        """)
        
        # Historical pattern
        st.markdown("### 📊 Historical Pattern")
        if current['anomaly_type'] == 'Duplicate Invoice':
            st.warning("⚠️ This vendor typically invoices once per month. This is the 2nd invoice this week.")
        elif current['anomaly_type'] == 'Unusual Vendor Pattern':
            st.warning("⚠️ New vendor with no history. First transaction is 5x higher than average new vendor.")
        else:
            st.warning("⚠️ User typically approves 5-10 transactions per day. 48 in 2 minutes is highly unusual.")
    
    with col2:
        st.markdown("### 🤖 AI Analysis")
        st.success(f"""
        **Confidence Score:** {current['risk_score']:.0f}%
        
        **Why Flagged:**
        {current.get('ai_explanation', 'Pattern anomaly detected')}
        
        **Recommended Action:**
        {'Reject and investigate' if current['risk_score'] > 85 else 'Request additional documentation'}
        """)
        
        # Show comparison for duplicates
        if current['anomaly_type'] == 'Duplicate Invoice':
            st.markdown("### 🔍 Comparison View")
            comparison_data = pd.DataFrame({
                'Field': ['Invoice ID', 'Vendor', 'Amount', 'Date'],
                'Current (INV-2947)': ['INV-2947', 'Acme Corp', '$15,420', 'Sept 14'],
                'Original (INV-2946)': ['INV-2946', 'Acme Corp', '$15,420', 'Sept 13']
            })
            st.dataframe(comparison_data, use_container_width=True)
    
    # Action buttons
    st.markdown("---")
    st.markdown("### Take Action")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("✅ APPROVE", type="primary", use_container_width=True):
            show_action_confirmation('approve', current)
    
    with col2:
        if st.button("❌ REJECT", type="primary", use_container_width=True):
            show_action_confirmation('reject', current)
    
    with col3:
        if st.button("📧 REQUEST INFO", type="primary", use_container_width=True):
            show_action_confirmation('request', current)

def show_action_confirmation(action, transaction):
    """Show confirmation dialog for actions"""
    st.markdown("---")
    st.markdown("### ⚡ Action Confirmation")
    
    action_text = {
        'approve': 'APPROVING',
        'reject': 'REJECTING',
        'request': 'REQUESTING INFO for'
    }
    
    st.info(f"You are {action_text[action]} Transaction {transaction['transaction_id']}")
    
    # Pre-filled reason
    reason = st.text_area(
        "Reason (AI Pre-filled):",
        value=f"{transaction['anomaly_type']} - {transaction.get('ai_explanation', '')}",
        height=100
    )
    
    # Additional notes
    notes = st.text_area("Additional Notes (Optional):", height=100)
    
    # Evidence capture notice
    st.success(f"""
    ✅ Evidence Automatically Captured:
    - Screenshot of analysis
    - Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    - Reviewer: Rachel Chen
    - AI Confidence Score: {transaction['risk_score']:.0f}%
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("✅ CONFIRM", type="primary", use_container_width=True):
            # Record the action
            action_record = {
                'timestamp': datetime.now(),
                'transaction_id': transaction['transaction_id'],
                'action': action,
                'reason': reason,
                'notes': notes,
                'reviewer': 'Rachel Chen',
                'ai_score': transaction['risk_score']
            }
            
            st.session_state.actions_taken.append(action_record)
            st.session_state.reviewed_items.append(transaction['transaction_id'])
            
            st.success(f"✅ Transaction {action}ed successfully!")
            st.balloons()
            
            # Return to dashboard after 2 seconds
            time.sleep(2)
            st.session_state.current_page = 'dashboard'
            st.rerun()
    
    with col2:
        if st.button("❌ CANCEL", use_container_width=True):
            st.rerun()

def show_evidence_repository():
    """Show audit trail and evidence repository"""
    st.title("📁 Evidence Repository")
    
    if st.button("← Back to Dashboard"):
        st.session_state.current_page = 'dashboard'
        st.rerun()
    
    st.markdown("---")
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    with col1:
        date_filter = st.selectbox("Time Period", ["Last 30 days", "Last 7 days", "Today"])
    with col2:
        user_filter = st.selectbox("Reviewer", ["All", "Rachel Chen", "John Smith"])
    with col3:
        action_filter = st.selectbox("Action Type", ["All", "Approved", "Rejected", "Info Requested"])
    
    # Display actions taken
    if st.session_state.actions_taken:
        st.markdown("### Recent Actions")
        for action in reversed(st.session_state.actions_taken):
            action_icon = {'approve': '✅', 'reject': '❌', 'request': '📧'}.get(action['action'], '❓')
            
            st.markdown(f"""
            <div class="exception-card low-risk">
                <strong>{action_icon} {action['timestamp'].strftime('%Y-%m-%d %H:%M')} | {action['action'].upper()}</strong><br>
                Transaction: {action['transaction_id']}<br>
                Reason: {action['reason']}<br>
                AI Score: {action['ai_score']:.0f}%<br>
                Evidence: Screenshot, Notes, Audit Log
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 5])
            with col1:
                st.button("📥 Download", key=f"dl_{action['transaction_id']}")
    else:
        st.info("No actions recorded yet. Review some exceptions to see the audit trail.")
    
    # Export options
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Export Audit Report", type="primary", use_container_width=True):
            st.success("Audit report exported successfully!")
    
    with col2:
        if st.button("📦 Download All Evidence", type="primary", use_container_width=True):
            st.success("Evidence package created successfully!")

# Main app navigation
def main():
    # Sidebar for navigation (hidden by default, can be toggled)
    with st.sidebar:
        st.markdown("## Navigation")
        if st.button("🏠 Dashboard"):
            st.session_state.current_page = 'dashboard'
        if st.button("📁 Evidence Repository"):
            st.session_state.current_page = 'evidence'
        
        st.markdown("---")
        st.markdown("### Quick Stats")
        st.metric("Reviews Today", len(st.session_state.reviewed_items))
        st.metric("Actions Taken", len(st.session_state.actions_taken))
        
        st.markdown("---")
        st.markdown("### Demo Mode")
        st.info("This is a prototype for customer validation. In production, this would connect to your ERP system.")
    
    # Main content area
    if st.session_state.current_page == 'dashboard':
        show_dashboard()
    elif st.session_state.current_page == 'review':
        show_review_screen()
    elif st.session_state.current_page == 'evidence':
        show_evidence_repository()

if __name__ == "__main__":
    main()
