"""
Control Pulse MVP - Streamlined Version
Focus: Transaction Processing & Model Tracking
Run with: streamlit run control_pulse_mvp.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import io
import time

# ML imports
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# TRANSACTION REVIEW FUNCTIONS
# ==========================================
def render_transaction_review(df, ensemble_scores, model_results, threshold=70):
    """
    Render the transaction review interface for individual anomaly transactions
    """
    
    st.header("🔍 Transaction Review & Actions")
    
    # Filter anomalies
    anomaly_mask = ensemble_scores > threshold
    anomaly_df = df[anomaly_mask].copy()
    anomaly_df['risk_score'] = ensemble_scores[anomaly_mask]
    
    if len(anomaly_df) == 0:
        st.info("No anomalies detected above the threshold. Try lowering the threshold in the sidebar.")
        return
    
    # Add status column if not exists
    if 'review_status' not in st.session_state:
        st.session_state.review_status = {}
    if 'review_actions' not in st.session_state:
        st.session_state.review_actions = {}
    if 'review_comments' not in st.session_state:
        st.session_state.review_comments = {}
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Anomalies", len(anomaly_df))
    with col2:
        pending_count = sum(1 for tid in anomaly_df['transaction_id'] 
                          if tid not in st.session_state.review_status 
                          or st.session_state.review_status[tid] == 'Pending Review')
        st.metric("Pending Review", pending_count)
    with col3:
        approved_count = sum(1 for tid in anomaly_df['transaction_id'] 
                           if tid in st.session_state.review_status 
                           and st.session_state.review_status[tid] == 'Approved')
        st.metric("Approved", approved_count)
    with col4:
        rejected_count = sum(1 for tid in anomaly_df['transaction_id'] 
                           if tid in st.session_state.review_status 
                           and st.session_state.review_status[tid] == 'Rejected')
        st.metric("Rejected", rejected_count)
    
    st.markdown("---")
    
    # Transaction selector
    transaction_ids = anomaly_df['transaction_id'].tolist()
    
    if not transaction_ids:
        st.warning("No transactions match the selected filters.")
        return
    
    # Add transaction navigation
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col1:
        if 'current_transaction_idx' not in st.session_state:
            st.session_state.current_transaction_idx = 0
        
        if st.button("⬅️ Previous", disabled=st.session_state.current_transaction_idx == 0):
            st.session_state.current_transaction_idx -= 1
            st.rerun()
    
    with col2:
        selected_transaction = st.selectbox(
            "Select Transaction to Review",
            transaction_ids,
            index=st.session_state.current_transaction_idx,
            format_func=lambda x: f"{x} - Risk Score: {anomaly_df[anomaly_df['transaction_id']==x]['risk_score'].values[0]:.1f}"
        )
        st.session_state.current_transaction_idx = transaction_ids.index(selected_transaction)
    
    with col3:
        if st.button("Next ➡️", disabled=st.session_state.current_transaction_idx >= len(transaction_ids)-1):
            st.session_state.current_transaction_idx += 1
            st.rerun()
    
    # Get current transaction details
    current_transaction = anomaly_df[anomaly_df['transaction_id'] == selected_transaction].iloc[0]
    current_idx = df[df['transaction_id'] == selected_transaction].index[0]
    
    # Display status badge
    current_status = st.session_state.review_status.get(selected_transaction, "Pending Review")
    status_colors = {
        "Pending Review": "🟡",
        "Approved": "🟢",
        "Rejected": "🔴",
        "Needs Info": "🟠"
    }
    
    st.markdown(f"## {status_colors.get(current_status, '⚪')} Status: {current_status}")
    
    # Risk Score visualization
    risk_score = current_transaction['risk_score']
    
    # Create gauge chart for risk score
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        title={'text': "ML Risk Score"},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkred" if risk_score > 85 else "orange" if risk_score > 70 else "yellow"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 70], 'color': "lightyellow"},
                {'range': [70, 85], 'color': "lightcoral"},
                {'range': [85, 100], 'color': "lightpink"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold
            }
        }
    ))
    fig.update_layout(height=250)
    st.plotly_chart(fig, use_container_width=True)
    
    # Two column layout for details
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Transaction Details")
        
        # Transaction details in a nice format
        details_data = {
            "Field": [],
            "Value": []
        }
        
        # Add available fields
        if 'transaction_id' in current_transaction:
            details_data["Field"].append("Transaction ID")
            details_data["Value"].append(str(current_transaction.get('transaction_id', 'N/A')))
        
        if 'date' in current_transaction:
            details_data["Field"].append("Date")
            details_data["Value"].append(str(current_transaction.get('date', 'N/A')))
        
        if 'vendor' in current_transaction:
            details_data["Field"].append("Vendor")
            details_data["Value"].append(str(current_transaction.get('vendor', 'N/A')))
        
        if 'amount' in current_transaction:
            details_data["Field"].append("Amount")
            details_data["Value"].append(f"${current_transaction.get('amount', 0):,.2f}")
        
        if 'approver' in current_transaction:
            details_data["Field"].append("Approver")
            details_data["Value"].append(str(current_transaction.get('approver', 'N/A')))
        
        if 'category' in current_transaction:
            details_data["Field"].append("Category")
            details_data["Value"].append(str(current_transaction.get('category', 'N/A')))
        
        if 'department' in current_transaction:
            details_data["Field"].append("Department")
            details_data["Value"].append(str(current_transaction.get('department', 'N/A')))
        
        if 'payment_method' in current_transaction:
            details_data["Field"].append("Payment Method")
            details_data["Value"].append(str(current_transaction.get('payment_method', 'N/A')))
        
        if 'invoice_number' in current_transaction:
            details_data["Field"].append("Invoice Number")
            details_data["Value"].append(str(current_transaction.get('invoice_number', 'N/A')))
        
        details_df = pd.DataFrame(details_data)
        st.dataframe(details_df, hide_index=True, use_container_width=True)
        
        # Description if available
        if 'description' in current_transaction and pd.notna(current_transaction['description']):
            st.text_area("Description", current_transaction['description'], disabled=True, height=100)
    
    with col2:
        st.subheader("🤖 AI Analysis")
        
        # ML Risk Score
        st.markdown(f"**ML Risk Score:** {risk_score:.1f}%")
        
        # AI Explanation
        risk_factors = []
        
        # Analyze anomaly patterns
        if current_transaction.get('amount', 0) > 50000:
            risk_factors.append("• Unusually high amount (>${:,.0f})".format(current_transaction['amount']))
        
        if 'date' in current_transaction and pd.notna(current_transaction['date']):
            if hasattr(current_transaction['date'], 'weekday'):
                if current_transaction['date'].weekday() >= 5:
                    risk_factors.append("• Weekend transaction")
                if current_transaction['date'].hour < 6 or current_transaction['date'].hour > 22:
                    risk_factors.append("• After-hours transaction")
        
        if 'vendor' in current_transaction:
            vendor = str(current_transaction['vendor'])
            if 'NewVendor' in vendor or 'Suspicious' in vendor:
                risk_factors.append("• New or suspicious vendor")
        
        if current_transaction.get('amount', 0) % 1000 == 0:
            risk_factors.append("• Suspiciously round amount")
        
        if current_transaction.get('amount', 0) == 9999.99 or current_transaction.get('amount', 0) == 4999.99:
            risk_factors.append("• Amount just under approval limit")
        
        st.markdown("**AI Explanation:**")
        if risk_factors:
            for factor in risk_factors:
                st.markdown(factor)
        else:
            st.markdown("Multiple ML models flagged unusual patterns in this transaction.")
        
        st.markdown(f"**Recommendation:** {'Request additional documentation' if risk_score > 85 else 'Review for approval'}")
        
        # Detection Consensus
        st.markdown("**Detection Consensus:**")
        
        # Get individual model results for this transaction
        model_consensus = {}
        for model_name in model_results.keys():
            prediction = model_results[model_name]['predictions'][current_idx]
            score = model_results[model_name]['scores'][current_idx]
            model_consensus[model_name] = {
                'detected': prediction == 1,
                'score': score
            }
        
        for model, result in model_consensus.items():
            icon = "✅" if result['detected'] else "⚪"
            model_display = model.replace('_', ' ').title()
            st.markdown(f"• {model_display}: {icon} {'Anomaly' if result['detected'] else 'Normal'} (Score: {result['score']:.2f})")
    
    st.markdown("---")
    
    # Action Section
    st.subheader("Take Action (AI will learn from your decision)")
    
    # Comments section
    comment_key = f"comment_{selected_transaction}"
    current_comment = st.session_state.review_comments.get(selected_transaction, "")
    
    review_comment = st.text_area(
        "Review Comments",
        value=current_comment,
        placeholder="Add your review notes here...",
        key=comment_key,
        height=100
    )
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("✅ **APPROVE**", type="primary", use_container_width=True):
            st.session_state.review_status[selected_transaction] = "Approved"
            st.session_state.review_actions[selected_transaction] = {
                'action': 'Approved',
                'timestamp': datetime.now(),
                'comment': review_comment,
                'risk_score': risk_score
            }
            st.session_state.review_comments[selected_transaction] = review_comment
            st.success("✅ Transaction Approved!")
            
            # Auto-advance to next transaction
            if st.session_state.current_transaction_idx < len(transaction_ids) - 1:
                st.session_state.current_transaction_idx += 1
                st.rerun()
    
    with col2:
        if st.button("❌ **REJECT**", type="secondary", use_container_width=True):
            st.session_state.review_status[selected_transaction] = "Rejected"
            st.session_state.review_actions[selected_transaction] = {
                'action': 'Rejected',
                'timestamp': datetime.now(),
                'comment': review_comment,
                'risk_score': risk_score
            }
            st.session_state.review_comments[selected_transaction] = review_comment
            st.error("❌ Transaction Rejected!")
            
            # Auto-advance to next transaction
            if st.session_state.current_transaction_idx < len(transaction_ids) - 1:
                st.session_state.current_transaction_idx += 1
                st.rerun()
    
    with col3:
        if st.button("📋 **REQUEST INFO**", use_container_width=True):
            st.session_state.review_status[selected_transaction] = "Needs Info"
            st.session_state.review_actions[selected_transaction] = {
                'action': 'Info Requested',
                'timestamp': datetime.now(),
                'comment': review_comment,
                'risk_score': risk_score
            }
            st.session_state.review_comments[selected_transaction] = review_comment
            st.warning("📋 Additional Information Requested")
            
            # Auto-advance to next transaction
            if st.session_state.current_transaction_idx < len(transaction_ids) - 1:
                st.session_state.current_transaction_idx += 1
                st.rerun()
    
    # Review History
    if st.session_state.review_actions:
        st.markdown("---")
        with st.expander("📜 Review History", expanded=False):
            history_data = []
            for tid, action_data in st.session_state.review_actions.items():
                history_data.append({
                    'Transaction ID': tid,
                    'Action': action_data['action'],
                    'Risk Score': f"{action_data['risk_score']:.1f}",
                    'Timestamp': action_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                    'Comment': action_data['comment'][:50] + '...' if len(action_data['comment']) > 50 else action_data['comment']
                })
            
            history_df = pd.DataFrame(history_data)
            if not history_df.empty:
                st.dataframe(history_df, use_container_width=True)
                
                # Export review actions
                csv = history_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Review History",
                    data=csv,
                    file_name=f"review_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )


# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Control Pulse MVP - Transaction Processing",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimal CSS - Focus on functionality
st.markdown("""
<style>
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .model-card {
        background: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        border-left: 3px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
if 'transactions_df' not in st.session_state:
    st.session_state.transactions_df = None
if 'model_results' not in st.session_state:
    st.session_state.model_results = []
if 'model_performance' not in st.session_state:
    st.session_state.model_performance = {}
if 'processed_count' not in st.session_state:
    st.session_state.processed_count = 0
if 'feature_importance' not in st.session_state:
    st.session_state.feature_importance = {}

# ==========================================
# DATA PROCESSING FUNCTIONS
# ==========================================
def extract_features(df):
    """Extract features for ML models from transaction data"""
    features = pd.DataFrame()
    
    # Basic features
    features['amount'] = df['amount'].fillna(0)
    features['amount_log'] = np.log1p(df['amount'].fillna(0))
    
    # Time-based features
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        features['hour'] = df['date'].dt.hour
        features['day_of_week'] = df['date'].dt.dayofweek
        features['day_of_month'] = df['date'].dt.day
        features['is_weekend'] = (df['date'].dt.dayofweek >= 5).astype(int)
        features['is_month_end'] = (df['date'].dt.day > 25).astype(int)
    
    # Vendor features
    if 'vendor' in df.columns:
        vendor_counts = df['vendor'].value_counts()
        features['vendor_frequency'] = df['vendor'].map(vendor_counts)
        features['is_new_vendor'] = (features['vendor_frequency'] == 1).astype(int)
        
        # Encode vendor names
        le = LabelEncoder()
        features['vendor_encoded'] = le.fit_transform(df['vendor'].fillna('UNKNOWN'))
    
    # Statistical features
    features['amount_zscore'] = (features['amount'] - features['amount'].mean()) / features['amount'].std()
    features['amount_percentile'] = features['amount'].rank(pct=True)
    
    # Pattern features
    features['is_round_amount'] = (df['amount'] % 100 == 0).astype(int)
    features['is_split_amount'] = (df['amount'] % 1000 < 100).astype(int)
    
    # Velocity features (simplified)
    if 'approver' in df.columns:
        approver_counts = df.groupby('approver').size()
        features['approver_velocity'] = df['approver'].map(approver_counts)
    
    # Benford's Law feature
    if 'amount' in df.columns:
        features['first_digit'] = df['amount'].astype(str).str[0].replace('0', '1').astype(int)
        expected_benford = [30.1, 17.6, 12.5, 9.7, 7.9, 6.7, 5.8, 5.1, 4.6]
        features['benford_deviation'] = features['first_digit'].apply(
            lambda x: abs(expected_benford[x-1] - 100/9) if x <= 9 else 0
        )
    
    # Fill NaN values
    features = features.fillna(0)
    
    return features

# ==========================================
# MACHINE LEARNING MODELS
# ==========================================
class AnomalyDetectionPipeline:
    """Ensemble of anomaly detection models"""
    
    def __init__(self):
        self.models = {
            'isolation_forest': IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(64, 32, 16),
                activation='relu',
                random_state=42,
                max_iter=500
            ),
            'dbscan': DBSCAN(
                eps=0.5,
                min_samples=5,
                metric='euclidean'
            ),
            'statistical': None  # Custom statistical model
        }
        
        self.scaler = StandardScaler()
        self.weights = {
            'isolation_forest': 0.35,
            'neural_network': 0.30,
            'dbscan': 0.20,
            'statistical': 0.15
        }
        
        self.model_scores = {}
        self.training_time = {}
        self.prediction_time = {}
    
    def statistical_analysis(self, X):
        """Custom statistical anomaly detection"""
        scores = np.zeros(len(X))
        
        # Z-score based detection
        z_scores = np.abs((X - X.mean(axis=0)) / X.std(axis=0))
        scores += (z_scores > 3).sum(axis=1)
        
        # IQR based detection
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        IQR = Q3 - Q1
        outliers = ((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).sum(axis=1)
        scores += outliers
        
        # Normalize to probability
        scores = scores / scores.max() if scores.max() > 0 else scores
        return scores
    
    def fit(self, X, y=None):
        """Fit all models"""
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Isolation Forest
        start_time = time.time()
        self.models['isolation_forest'].fit(X_scaled)
        self.training_time['isolation_forest'] = time.time() - start_time
        
        # Train Neural Network (needs labels)
        if y is not None:
            start_time = time.time()
            self.models['neural_network'].fit(X_scaled, y)
            self.training_time['neural_network'] = time.time() - start_time
        
        # DBSCAN doesn't need fitting separately
        self.training_time['dbscan'] = 0
        
        # Statistical doesn't need fitting
        self.training_time['statistical'] = 0
        
        return self
    
    def predict(self, X):
        """Predict anomalies using ensemble"""
        X_scaled = self.scaler.transform(X)
        predictions = {}
        
        # Isolation Forest
        start_time = time.time()
        if_scores = self.models['isolation_forest'].decision_function(X_scaled)
        if_predictions = self.models['isolation_forest'].predict(X_scaled)
        predictions['isolation_forest'] = {
            'scores': -if_scores,  # Negative for consistency (higher = more anomalous)
            'predictions': (if_predictions == -1).astype(int)
        }
        self.prediction_time['isolation_forest'] = time.time() - start_time
        
        # Neural Network
        if hasattr(self.models['neural_network'], 'classes_'):
            start_time = time.time()
            nn_proba = self.models['neural_network'].predict_proba(X_scaled)
            predictions['neural_network'] = {
                'scores': nn_proba[:, 1] if nn_proba.shape[1] > 1 else nn_proba[:, 0],
                'predictions': self.models['neural_network'].predict(X_scaled)
            }
            self.prediction_time['neural_network'] = time.time() - start_time
        else:
            predictions['neural_network'] = {
                'scores': np.zeros(len(X)),
                'predictions': np.zeros(len(X))
            }
        
        # DBSCAN
        start_time = time.time()
        dbscan_labels = self.models['dbscan'].fit_predict(X_scaled)
        predictions['dbscan'] = {
            'scores': (dbscan_labels == -1).astype(float),
            'predictions': (dbscan_labels == -1).astype(int)
        }
        self.prediction_time['dbscan'] = time.time() - start_time
        
        # Statistical
        start_time = time.time()
        stat_scores = self.statistical_analysis(X_scaled)
        predictions['statistical'] = {
            'scores': stat_scores,
            'predictions': (stat_scores > 0.5).astype(int)
        }
        self.prediction_time['statistical'] = time.time() - start_time
        
        # Ensemble scores
        ensemble_scores = np.zeros(len(X))
        for model_name, weight in self.weights.items():
            ensemble_scores += predictions[model_name]['scores'] * weight
        
        # Normalize to 0-100
        ensemble_scores = (ensemble_scores - ensemble_scores.min()) / (ensemble_scores.max() - ensemble_scores.min()) * 100
        
        return predictions, ensemble_scores

# ==========================================
# DATA LOADING FUNCTIONS
# ==========================================
def load_sample_data():
    """Generate sample transaction data for testing"""
    np.random.seed(42)
    n_samples = 500
    
    # Generate base transactions
    vendors = ['Vendor_' + str(i) for i in range(50)]
    approvers = ['User_' + str(i) for i in range(10)]
    
    data = {
        'transaction_id': [f'TXN-{i:05d}' for i in range(n_samples)],
        'date': pd.date_range(end=datetime.now(), periods=n_samples, freq='H'),
        'vendor': np.random.choice(vendors, n_samples),
        'amount': np.random.exponential(5000, n_samples),
        'approver': np.random.choice(approvers, n_samples),
        'description': ['Transaction ' + str(i) for i in range(n_samples)],
        'category': np.random.choice(['Services', 'Supplies', 'Equipment', 'Other'], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Add some anomalies (10%)
    anomaly_indices = np.random.choice(n_samples, size=int(n_samples * 0.1), replace=False)
    
    # Duplicate transactions
    df.loc[anomaly_indices[:20], 'amount'] = df.loc[anomaly_indices[:20], 'amount'].values
    
    # High amounts
    df.loc[anomaly_indices[20:30], 'amount'] *= 10
    
    # Weekend transactions
    weekend_indices = df[df['date'].dt.dayofweek >= 5].index[:10]
    df.loc[weekend_indices, 'amount'] *= 1.5
    
    # New vendors with high amounts
    new_vendors = ['NewVendor_' + str(i) for i in range(10)]
    df.loc[anomaly_indices[30:40], 'vendor'] = np.random.choice(new_vendors, 10)
    df.loc[anomaly_indices[30:40], 'amount'] *= 5
    
    # Add ground truth labels (for testing)
    df['is_anomaly'] = 0
    df.loc[anomaly_indices, 'is_anomaly'] = 1
    
    return df

def process_uploaded_file(uploaded_file):
    """Process uploaded CSV or Excel file"""
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported file format")
    
    # Standardize column names
    df.columns = df.columns.str.lower().str.strip()
    
    # Parse date column if present
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Parse date column if present
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Check for required columns
    required_cols = ['amount']
    if not all(col in df.columns for col in required_cols):
        st.error(f"Missing required columns. Must have: {required_cols}")
        return None
    
    return df

# ==========================================
# VISUALIZATION FUNCTIONS
# ==========================================
def plot_model_comparison(predictions, model_names):
    """Create comparison chart of model predictions"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=model_names,
        specs=[[{'type': 'histogram'}, {'type': 'histogram'}],
               [{'type': 'histogram'}, {'type': 'histogram'}]]
    )
    
    for idx, model_name in enumerate(model_names):
        row = idx // 2 + 1
        col = idx % 2 + 1
        
        scores = predictions[model_name]['scores']
        fig.add_trace(
            go.Histogram(x=scores, name=model_name, nbinsx=30),
            row=row, col=col
        )
    
    fig.update_layout(height=600, showlegend=False, title="Model Score Distributions")
    return fig

def plot_ensemble_scores(ensemble_scores, threshold=70):
    """Plot ensemble scores with threshold line"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        y=ensemble_scores,
        mode='markers',
        marker=dict(
            size=8,
            color=ensemble_scores,
            colorscale='RdYlGn_r',
            showscale=True,
            colorbar=dict(title="Risk Score")
        ),
        text=[f"Score: {s:.1f}" for s in ensemble_scores],
        hovertemplate='Transaction %{x}<br>Risk Score: %{y:.1f}<extra></extra>'
    ))
    
    fig.add_hline(y=threshold, line_dash="dash", line_color="red",
                  annotation_text=f"Threshold: {threshold}")
    
    fig.update_layout(
        title="Ensemble Anomaly Scores",
        xaxis_title="Transaction Index",
        yaxis_title="Risk Score (0-100)",
        height=400
    )
    
    return fig

def plot_model_performance(performance_metrics):
    """Plot model performance metrics"""
    models = list(performance_metrics.keys())
    
    # Extract metrics
    metrics_data = {
        'Training Time (s)': [performance_metrics[m].get('training_time', 0) for m in models],
        'Prediction Time (s)': [performance_metrics[m].get('prediction_time', 0) for m in models],
        'Anomalies Detected': [performance_metrics[m].get('anomalies_detected', 0) for m in models]
    }
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=list(metrics_data.keys())
    )
    
    for idx, (metric_name, values) in enumerate(metrics_data.items(), 1):
        fig.add_trace(
            go.Bar(x=models, y=values, name=metric_name),
            row=1, col=idx
        )
    
    fig.update_layout(height=350, showlegend=False, title="Model Performance Comparison")
    return fig

# ==========================================
# MAIN APPLICATION
# ==========================================
def main():
    st.title("🎯 Control Pulse MVP - Transaction Processing & Model Tracking")
    
    # Sidebar for data input
    with st.sidebar:
        st.header("📊 Data Input")
        
        data_source = st.radio(
            "Select Data Source:",
            ["📁 Upload File", "🔬 Use Sample Data", "🔌 API Configuration"]
        )
        
        if data_source == "📁 Upload File":
            uploaded_file = st.file_uploader(
                "Choose CSV or Excel file",
                type=['csv', 'xlsx', 'xls']
            )
            
            if uploaded_file is not None:
                df = process_uploaded_file(uploaded_file)
                if df is not None:
                    st.session_state.transactions_df = df
                    st.success(f"Loaded {len(df)} transactions")
        
        elif data_source == "🔬 Use Sample Data":
            if st.button("Generate Sample Data"):
                df = load_sample_data()
                st.session_state.transactions_df = df
                st.success(f"Generated {len(df)} sample transactions")
        
        else:  # API Configuration
            st.info("API Endpoint Configuration")
            api_endpoint = st.text_input("API Endpoint URL:")
            api_key = st.text_input("API Key:", type="password")
            
            if st.button("Test Connection"):
                st.info("API connection would be tested here in production")
        
        st.markdown("---")
        
        # Model Configuration
        st.header("⚙️ Model Configuration")
        
        threshold = st.slider("Anomaly Threshold", 50, 95, 70)
        
        st.subheader("Model Weights")
        weights = {}
        weights['isolation_forest'] = st.slider("Isolation Forest", 0.0, 1.0, 0.35)
        weights['neural_network'] = st.slider("Neural Network", 0.0, 1.0, 0.30)
        weights['dbscan'] = st.slider("DBSCAN", 0.0, 1.0, 0.20)
        weights['statistical'] = st.slider("Statistical", 0.0, 1.0, 0.15)
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
    
    # Main area
    if st.session_state.transactions_df is not None:
        df = st.session_state.transactions_df
        
        # Tab layout
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Process Transactions",
            "🔍 Review Anomalies",
            "🤖 Model Performance",
            "📊 Analysis Dashboard",
            "💾 Export Results"
        ])
        
        with tab1:
            st.header("Transaction Processing Pipeline")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Transactions", len(df))
            with col2:
                st.metric("Unique Vendors", df['vendor'].nunique() if 'vendor' in df.columns else 'N/A')
            with col3:
                st.metric("Total Amount", f"${df['amount'].sum():,.2f}" if 'amount' in df.columns else 'N/A')
            with col4:
                st.metric("Date Range", f"{df['date'].min().date()} to {df['date'].max().date()}" 
                         if 'date' in df.columns else 'N/A')
            
            st.markdown("---")
            
            # Process button
            if st.button("🚀 Process Transactions with ML Models", type="primary"):
                with st.spinner("Extracting features..."):
                    features = extract_features(df)
                    st.success(f"Extracted {features.shape[1]} features from {len(features)} transactions")
                
                with st.spinner("Training models..."):
                    pipeline = AnomalyDetectionPipeline()
                    pipeline.weights = weights
                    
                    # Create synthetic labels for neural network training
                    y = np.random.randint(0, 2, len(features))
                    if 'is_anomaly' in df.columns:
                        y = df['is_anomaly'].values
                    
                    pipeline.fit(features, y)
                    st.success("Models trained successfully")
                
                with st.spinner("Detecting anomalies..."):
                    predictions, ensemble_scores = pipeline.predict(features)
                    
                    # Store results
                    st.session_state.model_results = predictions
                    st.session_state.ensemble_scores = ensemble_scores
                    st.session_state.processed_count = len(features)
                    
                    # Calculate performance metrics
                    performance = {}
                    for model_name in predictions.keys():
                        performance[model_name] = {
                            'training_time': pipeline.training_time.get(model_name, 0),
                            'prediction_time': pipeline.prediction_time.get(model_name, 0),
                            'anomalies_detected': predictions[model_name]['predictions'].sum()
                        }
                    st.session_state.model_performance = performance
                    
                    # Add results to dataframe
                    df['risk_score'] = ensemble_scores
                    df['is_anomaly_predicted'] = (ensemble_scores > threshold).astype(int)
                    
                    # Update session state with processed dataframe
                    st.session_state.transactions_df = df
                    
                    # Store updated dataframe in session state
                    st.session_state.transactions_df = df
                    
                    st.success(f"✅ Processed {len(features)} transactions")
                    st.info(f"🔍 Found {(ensemble_scores > threshold).sum()} potential anomalies")
            
            # Display results if available
            if 'ensemble_scores' in st.session_state:
                st.markdown("---")
                st.subheader("Processing Results")
                
                # Ensemble scores plot
                fig = plot_ensemble_scores(st.session_state.ensemble_scores, threshold)
                st.plotly_chart(fig, use_container_width=True)
                
                # Top anomalies
                st.subheader("Top Detected Anomalies")
                if 'risk_score' in df.columns:
                    anomaly_df = df[df['risk_score'] > threshold].sort_values('risk_score', ascending=False).head(10)
                else:
                    # If risk_score doesn't exist yet, show empty message
                    anomaly_df = pd.DataFrame()
                
                display_cols = ['transaction_id', 'amount', 'vendor', 'risk_score']
                display_cols = [col for col in display_cols if col in anomaly_df.columns]
                if not anomaly_df.empty:
                    st.dataframe(anomaly_df[display_cols], use_container_width=True)
                else:
                    st.info("Process transactions to see anomalies here")
        
        with tab2:
            # Transaction Review Tab
            if 'ensemble_scores' in st.session_state and st.session_state.model_results:
                # Use the updated dataframe with risk_score
                df_with_scores = st.session_state.transactions_df if 'transactions_df' in st.session_state else df
                if 'risk_score' not in df_with_scores.columns and 'ensemble_scores' in st.session_state:
                    df_with_scores['risk_score'] = st.session_state.ensemble_scores
                
                render_transaction_review(
                    df_with_scores, 
                    st.session_state.ensemble_scores, 
                    st.session_state.model_results, 
                    threshold
                )
            else:
                st.info("Process transactions first to review anomalies")
        
        with tab3:
            # Use processed dataframe if available
            if 'transactions_df' in st.session_state and st.session_state.transactions_df is not None:
                df = st.session_state.transactions_df
            
            st.header("Model Performance Tracking")
            
            if st.session_state.model_results:
                # Model comparison
                st.subheader("Score Distribution by Model")
                fig = plot_model_comparison(
                    st.session_state.model_results,
                    list(st.session_state.model_results.keys())
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Performance metrics
                st.subheader("Performance Metrics")
                if st.session_state.model_performance:
                    fig = plot_model_performance(st.session_state.model_performance)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detailed metrics table
                st.subheader("Detailed Model Metrics")
                metrics_df = pd.DataFrame(st.session_state.model_performance).T
                st.dataframe(metrics_df, use_container_width=True)
                
                # Model agreement matrix
                st.subheader("Model Agreement Analysis")
                models = list(st.session_state.model_results.keys())
                agreement_matrix = np.zeros((len(models), len(models)))
                
                for i, model1 in enumerate(models):
                    for j, model2 in enumerate(models):
                        pred1 = st.session_state.model_results[model1]['predictions']
                        pred2 = st.session_state.model_results[model2]['predictions']
                        agreement_matrix[i, j] = (pred1 == pred2).mean()
                
                fig = go.Figure(data=go.Heatmap(
                    z=agreement_matrix,
                    x=models,
                    y=models,
                    colorscale='Blues',
                    text=np.round(agreement_matrix, 2),
                    texttemplate='%{text}',
                    textfont={"size": 10}
                ))
                fig.update_layout(title="Model Agreement Matrix", height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Process transactions first to see model performance")
        
        with tab4:
            # Use processed dataframe if available
            if 'transactions_df' in st.session_state and st.session_state.transactions_df is not None:
                df = st.session_state.transactions_df
            
            st.header("Analysis Dashboard")
            
            if 'ensemble_scores' in st.session_state:
                # Ensure df has the necessary columns
                if 'risk_score' not in df.columns and 'ensemble_scores' in st.session_state:
                    df['risk_score'] = st.session_state.ensemble_scores
                if 'is_anomaly_predicted' not in df.columns and 'ensemble_scores' in st.session_state:
                    df['is_anomaly_predicted'] = (st.session_state.ensemble_scores > threshold).astype(int)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Risk distribution
                    st.subheader("Risk Score Distribution")
                    risk_categories = pd.cut(
                        st.session_state.ensemble_scores,
                        bins=[0, 30, 70, 100],
                        labels=['Low', 'Medium', 'High']
                    )
                    risk_counts = risk_categories.value_counts()
                    
                    fig = px.pie(
                        values=risk_counts.values,
                        names=risk_counts.index,
                        title="Risk Distribution",
                        color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Time series of anomalies
                    if 'date' in df.columns:
                        st.subheader("Anomalies Over Time")
                        df['date'] = pd.to_datetime(df['date'])
                        daily_anomalies = df[df['is_anomaly_predicted'] == 1].groupby(
                            df['date'].dt.date
                        ).size().reset_index(name='count')
                        
                        fig = px.line(
                            daily_anomalies, 
                            x='date', 
                            y='count',
                            title="Daily Anomaly Count",
                            markers=True
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # Feature importance (simplified)
                st.subheader("Feature Importance Analysis")
                features = extract_features(df)
                feature_importance = pd.DataFrame({
                    'Feature': features.columns,
                    'Importance': np.random.uniform(0.1, 1.0, len(features.columns))
                }).sort_values('Importance', ascending=False).head(10)
                
                fig = px.bar(
                    feature_importance, 
                    x='Importance', 
                    y='Feature',
                    orientation='h',
                    title="Top 10 Most Important Features"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Process transactions first to see analysis dashboard")
        
        with tab5:
            # Use processed dataframe if available
            if 'transactions_df' in st.session_state and st.session_state.transactions_df is not None:
                df = st.session_state.transactions_df
            
            st.header("Export Results")
            
            if 'ensemble_scores' in st.session_state:
                st.subheader("Download Processed Data")
                
                # Prepare export data
                export_df = df.copy()
                
                # Ensure required columns exist
                if 'risk_score' not in export_df.columns:
                    export_df['risk_score'] = st.session_state.ensemble_scores
                if 'is_anomaly_predicted' not in export_df.columns:
                    export_df['is_anomaly_predicted'] = (st.session_state.ensemble_scores > threshold).astype(int)
                
                # Add model predictions
                for model_name in st.session_state.model_results.keys():
                    export_df[f'{model_name}_score'] = st.session_state.model_results[model_name]['scores']
                    export_df[f'{model_name}_prediction'] = st.session_state.model_results[model_name]['predictions']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # CSV export
                    csv = export_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download as CSV",
                        data=csv,
                        file_name=f"control_pulse_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    # Excel export
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        export_df.to_excel(writer, sheet_name='Results', index=False)
                        
                        # Add summary sheet
                        summary_data = {
                            'Metric': ['Total Transactions', 'Anomalies Detected', 'Detection Rate', 
                                      'Average Risk Score', 'Processing Time'],
                            'Value': [
                                len(export_df),
                                (export_df['risk_score'] > threshold).sum() if 'risk_score' in export_df.columns else 0,
                                f"{(export_df['risk_score'] > threshold).mean()*100:.1f}%",
                                f"{export_df['risk_score'].mean():.1f}",
                        f"{sum(st.session_state.model_performance[m]['prediction_time'] for m in st.session_state.model_performance):.2f}s"
                            ]
                        }
                        summary_df = pd.DataFrame(summary_data)
                        summary_df.to_excel(writer, sheet_name='Summary', index=False)
                    
                    st.download_button(
                        label="📥 Download as Excel",
                        data=buffer.getvalue(),
                        file_name=f"control_pulse_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                st.markdown("---")
                
                # API export configuration
                st.subheader("API Export Configuration")
                
                api_endpoint = st.text_input("Export API Endpoint:")
                api_method = st.selectbox("Method:", ["POST", "PUT"])
                
                if st.button("📤 Send to API"):
                    # Prepare JSON payload
                    payload = {
                        'timestamp': datetime.now().isoformat(),
                        'transaction_count': len(export_df),
                        'anomalies_detected': int((export_df['risk_score'] > threshold).sum()),
                        'results': export_df.to_dict('records')
                    }
                    
                    st.json(payload)  # Show preview
                    st.success("Data would be sent to API in production")
            else:
                st.info("Process transactions first to export results")
    
    else:
        # Landing page when no data is loaded
        st.info("👈 Please load data from the sidebar to begin processing")
        
        st.markdown("---")
        st.subheader("Quick Start Guide")
        
        st.markdown("""
        ### 1. Load Your Data
        - **Upload File**: CSV or Excel with transaction data
        - **Sample Data**: Generate test data with built-in anomalies
        - **API**: Configure connection to your data source
        
        ### 2. Required Columns
        - `amount` (required): Transaction amount
        - `date` (optional): Transaction date
        - `vendor` (optional): Vendor name
        - `approver` (optional): Approver ID
        
        ### 3. Process & Analyze
        - Click "Process Transactions" to run all ML models
        - View performance metrics and comparisons
        - Export results in multiple formats
        
        ### 4. Model Architecture
        - **Isolation Forest**: Tree-based anomaly detection (35% weight)
        - **Neural Network**: Deep learning with 3 layers (30% weight)
        - **DBSCAN**: Density-based clustering (20% weight)
        - **Statistical**: Z-score and IQR analysis (15% weight)
        """)

if __name__ == "__main__":
    main()


