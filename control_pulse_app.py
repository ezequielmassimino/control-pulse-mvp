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

# Mobile-friendly CSS with responsive design
st.markdown("""
<style>
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
    
    h1 {
        font-size: 1.5rem !important;
    }
    
    h2 {
        font-size: 1.2rem !important;
    }
    
    h3 {
        font-size: 1rem !important;
    }
    
    .stMetric {
        padding: 0.5rem !important;
    }
    
    /* Stack columns on mobile */
    [data-testid="column"] {
        width: 100% !important;
        flex: 100% !important;
    }
}

/* Desktop and general styles */
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
    word-wrap: break-word;
}
.high-risk {border-left-color: #ff4444;}
.medium-risk {border-left-color: #ffaa00;}
.low-risk {border-left-color: #00c851;}

/* Improve mobile menu */
.st-emotion-cache-16idsys p {
    font-size: 0.9rem;
}

/* Better mobile data display */
.dataframe {
    font-size: 0.8rem !important;
}

/* Responsive charts */
.js-plotly-plot {
    max-width: 100%;
    overflow-x: auto;
}
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

# Load and prepare data with more exception types
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
    
    for i in range(40):  # Reduced to make room for more exceptions
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

def is_mobile():
    """Check if user is on mobile device - simplified approach"""
    # This is a simplified check. In production, you might use more sophisticated detection
    return st.session_state.get('mobile_view', False)

def show_dashboard():
    """Main dashboard view with mobile responsiveness"""
    st.title("🎯 Control Pulse")
    st.markdown("##### Financial Control Monitoring System")
    
    # User greeting - simplified for mobile
    col1, col2 = st.columns([2, 1]) if not is_mobile() else st.columns([1])
    with col1:
        st.markdown("### Good morning, Rachel!")
        st.markdown("You have **10 exceptions** requiring review")
    if not is_mobile():
        with col2:
            if st.button("👤 Settings"):
                st.info("Settings page (coming soon)")
    
    # Load data and calculate metrics
    df = load_data()
    metrics = calculate_metrics(df)
    exceptions = df[df['risk_score'] > 70].sort_values('risk_score', ascending=False)
    
    # Display key metrics - responsive grid
    st.markdown("---")
    
    # For mobile, show metrics in 2x2 grid instead of 1x4
    if is_mobile():
        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)
    else:
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
            label="✅ Reviewed",
            value=metrics['reviewed'],
            delta="+12"
        )
    
    with col3:
        st.metric(
            label="💰 Saved",
            value=f"${metrics['savings']/1000:.0f}K",
            delta="+23%"
        )
    
    with col4:
        st.metric(
            label="📊 Effectiveness",
            value=f"{metrics['effectiveness']}%",
            delta="+2%"
        )
    
    # Exception type filter
    st.markdown("---")
    st.markdown("### 🎯 Exceptions Requiring Review")
    
    # Add filter for exception types
    exception_types = ['All'] + list(exceptions['anomaly_type'].unique())
    selected_type = st.selectbox("Filter by Exception Type:", exception_types, index=0)
    
    # Filter exceptions based on selection
    if selected_type != 'All':
        filtered_exceptions = exceptions[exceptions['anomaly_type'] == selected_type]
    else:
        filtered_exceptions = exceptions
    
    st.markdown(f"*Showing {len(filtered_exceptions)} exception(s)*")
    
    # Display exceptions - mobile-friendly cards
    for idx, row in filtered_exceptions.iterrows():
        risk_class = 'high-risk' if row['risk_score'] > 85 else 'medium-risk' if row['risk_score'] > 75 else 'low-risk'
        risk_icon = '🔴' if row['risk_score'] > 85 else '🟡' if row['risk_score'] > 75 else '🟠'
        
        # Single column layout on mobile
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
    
    # Quick stats visualization - stack on mobile
    st.markdown("---")
    
    if is_mobile():
        # Stack charts vertically on mobile
        st.markdown("### 📈 Weekly Trend")
        dates = pd.date_range(end=datetime.now(), periods=7)
        trend_data = pd.DataFrame({
            'Date': dates,
            'Exceptions': np.random.randint(8, 15, 7),
            'Reviewed': np.random.randint(40, 60, 7)
        })
        
        fig = px.line(trend_data, x='Date', y=['Exceptions', 'Reviewed'],
                     title="Exception Detection Trend",
                     color_discrete_map={'Exceptions': '#ff4444', 'Reviewed': '#00c851'})
        fig.update_layout(height=250, showlegend=True, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 🎯 Risk Distribution")
        risk_dist = pd.DataFrame({
            'Risk Level': ['High', 'Medium', 'Low'],
            'Count': [4, 4, 42]
        })
        
        fig = px.pie(risk_dist, values='Count', names='Risk Level',
                    color_discrete_map={'High': '#ff4444', 'Medium': '#ffaa00', 'Low': '#00c851'},
                    title="Current Risk Distribution")
        fig.update_layout(height=250, margin=dict(l=0, r=0, t=30, b=0))
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
                         color_discrete_map={'Exceptions': '#ff4444', 'Reviewed': '#00c851'})
            fig.update_layout(height=300, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🎯 Risk Distribution")
            risk_dist = pd.DataFrame({
                'Risk Level': ['High', 'Medium', 'Low'],
                'Count': [4, 4, 42]
            })
            
            fig = px.pie(risk_dist, values='Count', names='Risk Level',
                        color_discrete_map={'High': '#ff4444', 'Medium': '#ffaa00', 'Low': '#00c851'},
                        title="Current Risk Distribution")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

def show_review_screen():
    """Detailed review screen for exceptions - mobile optimized"""
    df = load_data()
    exceptions = df[df['risk_score'] > 70].sort_values('risk_score', ascending=False)
    
    if st.session_state.current_exception >= len(exceptions):
        st.session_state.current_exception = 0
    
    current = exceptions.iloc[st.session_state.current_exception]
    
    # Simplified header for mobile
    if is_mobile():
        if st.button("← Back"):
            st.session_state.current_page = 'dashboard'
            st.rerun()
        st.markdown(f"### Exception {st.session_state.current_exception + 1}/{len(exceptions)}")
        if st.button("Next →"):
            st.session_state.current_exception = (st.session_state.current_exception + 1) % len(exceptions)
            st.rerun()
    else:
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
    risk_icon = '🔴' if current['risk_score'] > 85 else '🟡' if current['risk_score'] > 75 else '🟠'
    st.markdown(f"# {risk_icon} {current['anomaly_type']}")
    
    # Stack on mobile, side-by-side on desktop
    if is_mobile():
        # Mobile: Stack everything vertically
        st.markdown("### 📋 Transaction Details")
        st.info(f"""
        **ID:** {current['transaction_id']}
        **Vendor:** {current['vendor_name']}
        **Amount:** ${current['amount']:,.2f}
        **Approver:** {current['approver']}
        **Date:** {current['approval_date'].strftime('%Y-%m-%d')}
        """)
        
        st.markdown("### 🤖 AI Analysis")
        st.success(f"""
        **Risk Score:** {current['risk_score']:.0f}%
        
        **Why Flagged:**
        {current.get('ai_explanation', 'Pattern anomaly detected')}
        
        **Recommended:**
        {'Reject and investigate' if current['risk_score'] > 85 else 'Request additional documentation'}
        """)
        
        # Exception-specific warnings
        st.markdown("### ⚠️ Pattern Analysis")
        if current['anomaly_type'] == 'Duplicate Invoice':
            st.warning("This vendor typically invoices once per month. This is the 2nd invoice this week.")
        elif current['anomaly_type'] == 'Unusual Vendor Pattern':
            st.warning("New vendor with no history. First transaction is 5x higher than average.")
        elif current['anomaly_type'] == 'Split Invoice Pattern':
            st.warning("Multiple invoices just below approval threshold detected.")
        elif current['anomaly_type'] == 'Round Amount Anomaly':
            st.warning("Suspiciously round amount without detailed breakdown.")
        elif current['anomaly_type'] == 'Weekend Processing':
            st.warning("Transaction processed during unusual hours/weekend.")
        elif current['anomaly_type'] == 'Vendor Detail Change':
            st.warning("Recent change to vendor banking details detected.")
        elif current['anomaly_type'] == 'Geographic Anomaly':
            st.warning("Vendor location or jurisdiction has changed.")
        else:
            st.warning("Unusual pattern detected requiring review.")
    else:
        # Desktop: Two-column layout
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
            elif current['anomaly_type'] == 'Split Invoice Pattern':
                st.warning("⚠️ Multiple invoices just below approval threshold detected.")
            elif current['anomaly_type'] == 'Round Amount Anomaly':
                st.warning("⚠️ Suspiciously round amount without detailed breakdown.")
            elif current['anomaly_type'] == 'Weekend Processing':
                st.warning("⚠️ Transaction processed during unusual hours/weekend.")
            elif current['anomaly_type'] == 'Vendor Detail Change':
                st.warning("⚠️ Recent change to vendor banking details detected.")
            elif current['anomaly_type'] == 'Geographic Anomaly':
                st.warning("⚠️ Vendor location or jurisdiction has changed to high-risk area.")
            else:
                st.warning("⚠️ Unusual pattern detected in transaction history.")
        
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
                    'Current': ['INV-2947', 'Acme Corp', '$15,420', 'Sept 18'],
                    'Original': ['INV-2946', 'Acme Corp', '$15,420', 'Sept 17']
                })
                st.dataframe(comparison_data, use_container_width=True)
    
    # Action buttons - responsive layout
    st.markdown("---")
    st.markdown("### Take Action")
    
    if is_mobile():
        # Stack buttons on mobile
        if st.button("✅ APPROVE", type="primary", use_container_width=True):
            show_action_confirmation('approve', current)
        if st.button("❌ REJECT", type="primary", use_container_width=True):
            show_action_confirmation('reject', current)
        if st.button("📧 REQUEST INFO", type="primary", use_container_width=True):
            show_action_confirmation('request', current)
    else:
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
    """Show confirmation dialog for actions - mobile optimized"""
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
    
    # Mobile-responsive button layout
    if is_mobile():
        if st.button("✅ CONFIRM", type="primary", use_container_width=True):
            handle_action_confirmation(action, transaction, reason, notes)
        if st.button("❌ CANCEL", use_container_width=True):
            st.rerun()
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("✅ CONFIRM", type="primary", use_container_width=True):
                handle_action_confirmation(action, transaction, reason, notes)
        
        with col2:
            if st.button("❌ CANCEL", use_container_width=True):
                st.rerun()

def handle_action_confirmation(action, transaction, reason, notes):
    """Handle the confirmation of an action"""
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

def show_evidence_repository():
    """Show audit trail and evidence repository - mobile optimized"""
    st.title("📁 Evidence Repository")
    
    if st.button("← Back to Dashboard"):
        st.session_state.current_page = 'dashboard'
        st.rerun()
    
    st.markdown("---")
    
    # Filter options - stack on mobile
    if is_mobile():
        date_filter = st.selectbox("Time Period", ["Last 30 days", "Last 7 days", "Today"])
        user_filter = st.selectbox("Reviewer", ["All", "Rachel Chen", "John Smith"])
        action_filter = st.selectbox("Action Type", ["All", "Approved", "Rejected", "Info Requested"])
    else:
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
            
            if not is_mobile():
                col1, col2 = st.columns([1, 5])
                with col1:
                    st.button("📥 Download", key=f"dl_{action['transaction_id']}_{action['timestamp']}")
            else:
                st.button("📥 Download Evidence", key=f"dl_{action['transaction_id']}_{action['timestamp']}", use_container_width=True)
    else:
        st.info("No actions recorded yet. Review some exceptions to see the audit trail.")
    
    # Export options - mobile responsive
    st.markdown("---")
    
    if is_mobile():
        if st.button("📊 Export Audit Report", type="primary", use_container_width=True):
            st.success("Audit report exported successfully!")
        if st.button("📦 Download All Evidence", type="primary", use_container_width=True):
            st.success("Evidence package created successfully!")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Export Audit Report", type="primary", use_container_width=True):
                st.success("Audit report exported successfully!")
        
        with col2:
            if st.button("📦 Download All Evidence", type="primary", use_container_width=True):
                st.success("Evidence package created successfully!")

# Main app navigation
def main():
    # Check if mobile view toggle exists
    mobile_toggle = st.checkbox("📱 Mobile View", value=False, key="mobile_view")
    
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
        st.markdown("### Exception Types")
        st.markdown("""
        - 🔴 **High Risk** (85%+)
        - 🟡 **Medium Risk** (75-84%)
        - 🟠 **Low-Medium** (70-74%)
        """)
        
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
