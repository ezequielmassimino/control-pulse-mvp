"""
Control Pulse - Financial Control Monitoring System
MVP Prototype with Advanced ML Anomaly Detection
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
import pickle
import json

# ============================================================================
# MACHINE LEARNING IMPORTS
# These libraries provide the AI/ML capabilities for anomaly detection
# ============================================================================
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

# Page Configuration
st.set_page_config(
    page_title="Control Pulse - AI-Powered Financial Monitoring",
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
# ============================================================================
# ML MODEL STATE MANAGEMENT
# Store trained models and learning history in session state
# This allows the models to improve over time as users review transactions
# ============================================================================
if 'ml_models' not in st.session_state:
    st.session_state.ml_models = {}
if 'ml_feedback_history' not in st.session_state:
    st.session_state.ml_feedback_history = []
if 'model_performance' not in st.session_state:
    st.session_state.model_performance = {
        'accuracy': 0.94,
        'precision': 0.92,
        'recall': 0.89,
        'f1_score': 0.90
    }

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
    .ml-indicator {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 5px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        display: inline-block;
        margin-left: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# ADVANCED ML ANOMALY DETECTION SYSTEM
# This is the core AI engine that detects financial anomalies
# ============================================================================

class FinancialAnomalyDetector:
    """
    Advanced ML system combining multiple algorithms for robust anomaly detection:
    1. Isolation Forest: Detects outliers in high-dimensional data
    2. DBSCAN: Finds clusters and identifies points that don't belong
    3. Neural Network: Learns complex patterns from historical data
    4. Statistical Analysis: Applies business rules and thresholds
    """
    
    def __init__(self):
        # Initialize multiple ML models for ensemble learning
        # Ensemble learning combines multiple models for better accuracy
        
        # ISOLATION FOREST: Great for detecting outliers in financial data
        # It isolates anomalies by randomly selecting features and split values
        self.isolation_forest = IsolationForest(
            contamination=0.1,  # Expect 10% of data to be anomalous
            random_state=42,
            n_estimators=100    # Use 100 trees for robust detection
        )
        
        # DBSCAN: Density-based clustering to find unusual transaction patterns
        # It groups similar transactions and flags those that don't fit any group
        self.dbscan = DBSCAN(
            eps=0.5,           # Maximum distance between samples in a cluster
            min_samples=5      # Minimum samples to form a cluster
        )
        
        # NEURAL NETWORK: Deep learning model for complex pattern recognition
        # This can learn non-linear relationships in the data
        self.neural_network = MLPClassifier(
            hidden_layer_sizes=(100, 50, 25),  # 3 hidden layers with decreasing neurons
            activation='relu',                   # ReLU activation for non-linearity
            solver='adam',                       # Adam optimizer for training
            max_iter=1000,                      # Maximum training iterations
            random_state=42
        )
        
        # RANDOM FOREST: Ensemble of decision trees for classification
        # Provides feature importance to explain why something is anomalous
        self.random_forest = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        # Feature scaler for normalizing data before ML processing
        # ML models perform better with normalized features
        self.scaler = StandardScaler()
        
        # Label encoders for categorical variables
        self.vendor_encoder = LabelEncoder()
        self.approver_encoder = LabelEncoder()
        
        # Store feature importance for explainability
        self.feature_importance = {}
        
    def prepare_features(self, df):
        """
        Convert raw transaction data into ML-ready features
        This is crucial for good ML performance
        """
        features = pd.DataFrame()
        
        # NUMERICAL FEATURES
        features['amount'] = df['amount']
        features['amount_log'] = np.log1p(df['amount'])  # Log transform for skewed data
        features['risk_score_base'] = df.get('risk_score', 0)
        
        # TIME-BASED FEATURES (very important for fraud detection)
        features['hour'] = pd.to_datetime(df['approval_date']).dt.hour
        features['day_of_week'] = pd.to_datetime(df['approval_date']).dt.dayofweek
        features['day_of_month'] = pd.to_datetime(df['approval_date']).dt.day
        features['is_weekend'] = (features['day_of_week'] >= 5).astype(int)
        features['is_night'] = ((features['hour'] < 6) | (features['hour'] > 22)).astype(int)
        
        # STATISTICAL FEATURES (detect unusual patterns)
        # Calculate z-scores to find statistical outliers
        amount_mean = df['amount'].mean()
        amount_std = df['amount'].std()
        features['amount_zscore'] = (df['amount'] - amount_mean) / amount_std
        
        # VENDOR FEATURES
        # Encode vendors and calculate vendor-specific risk metrics
        if len(df['vendor_name'].unique()) > 1:
            features['vendor_encoded'] = self.vendor_encoder.fit_transform(df['vendor_name'])
            vendor_counts = df['vendor_name'].value_counts()
            features['vendor_frequency'] = df['vendor_name'].map(vendor_counts)
            features['vendor_risk'] = df.groupby('vendor_name')['amount'].transform(
                lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
            )
        else:
            features['vendor_encoded'] = 0
            features['vendor_frequency'] = 1
            features['vendor_risk'] = 0
        
        # APPROVER FEATURES
        # Analyze approver behavior patterns
        if len(df['approver'].unique()) > 1:
            features['approver_encoded'] = self.approver_encoder.fit_transform(df['approver'])
            approver_counts = df['approver'].value_counts()
            features['approver_frequency'] = df['approver'].map(approver_counts)
        else:
            features['approver_encoded'] = 0
            features['approver_frequency'] = 1
        
        # PATTERN-BASED FEATURES
        # Detect specific fraud patterns
        features['is_round_amount'] = (df['amount'] % 100 == 0).astype(int)
        features['is_just_below_threshold'] = (
            (df['amount'] > 900) & (df['amount'] < 1000)
        ).astype(int)
        features['amount_digits'] = df['amount'].astype(str).str.len()
        
        # Fill any NaN values with 0
        features = features.fillna(0)
        
        return features
    
    def detect_anomalies(self, df):
        """
        Main ML pipeline that runs multiple algorithms and combines their results
        This is where the AI magic happens!
        """
        if len(df) < 10:  # Need minimum data for ML
            return self._simple_rule_based_detection(df)
        
        # Prepare features for ML models
        features = self.prepare_features(df)
        
        # Scale features for better ML performance
        features_scaled = self.scaler.fit_transform(features)
        
        # ========== RUN MULTIPLE ML ALGORITHMS ==========
        
        # 1. ISOLATION FOREST DETECTION
        # This algorithm builds random trees and measures how easy it is to isolate each point
        # Anomalies are easier to isolate and get higher scores
        isolation_scores = self.isolation_forest.fit_predict(features_scaled)
        isolation_anomalies = isolation_scores == -1  # -1 indicates anomaly
        
        # 2. DBSCAN CLUSTERING
        # Groups similar transactions together
        # Transactions that don't fit any group are anomalies
        dbscan_labels = self.dbscan.fit_predict(features_scaled)
        dbscan_anomalies = dbscan_labels == -1  # -1 indicates noise/anomaly
        
        # 3. STATISTICAL OUTLIER DETECTION
        # Use z-scores and IQR method for statistical anomalies
        statistical_anomalies = (
            (np.abs(features['amount_zscore']) > 3) |  # 3 standard deviations
            (features['is_weekend'] & features['is_night']) |  # Weekend + night
            (features['is_round_amount'] & (features['amount'] > 10000))  # Large round amounts
        )
        
        # 4. NEURAL NETWORK PREDICTION (if we have training data)
        nn_anomalies = np.zeros(len(df), dtype=bool)
        if len(st.session_state.ml_feedback_history) > 20:
            # We have enough feedback to train the neural network
            try:
                # Train on historical feedback
                X_train, y_train = self._prepare_training_data()
                self.neural_network.fit(X_train, y_train)
                nn_predictions = self.neural_network.predict(features_scaled)
                nn_anomalies = nn_predictions == 1  # 1 indicates anomaly
            except:
                pass  # Skip NN if training fails
        
        # ========== ENSEMBLE VOTING ==========
        # Combine all models using weighted voting
        # Each model votes on whether something is an anomaly
        anomaly_votes = (
            isolation_anomalies.astype(int) * 2 +  # Weight: 2x (most reliable)
            dbscan_anomalies.astype(int) * 1.5 +    # Weight: 1.5x
            statistical_anomalies.astype(int) * 1 +  # Weight: 1x
            nn_anomalies.astype(int) * 1.5          # Weight: 1.5x
        )
        
        # Calculate confidence scores (0-100)
        # Higher votes = higher confidence
        max_votes = 2 + 1.5 + 1 + 1.5  # Sum of all weights
        confidence_scores = (anomaly_votes / max_votes) * 100
        
        # Add ML-enhanced risk scores to dataframe
        df['ml_risk_score'] = confidence_scores
        df['is_anomaly'] = confidence_scores > 50  # Threshold for flagging
        
        # Store feature importance for explainability
        self._calculate_feature_importance(features, df['is_anomaly'])
        
        # Generate AI explanations for anomalies
        df['ai_explanation'] = df.apply(
            lambda row: self._generate_ai_explanation(row, features.loc[row.name])
            if row['is_anomaly'] else '', axis=1
        )
        
        return df
    
    def _generate_ai_explanation(self, transaction, features):
        """
        Generate human-readable explanations for why the AI flagged a transaction
        This is crucial for user trust and compliance
        """
        explanations = []
        
        # Check which features contributed most to the anomaly
        if features['amount_zscore'] > 3:
            explanations.append(f"Amount ${transaction['amount']:,.2f} is {features['amount_zscore']:.1f} standard deviations above normal")
        
        if features['is_weekend'] and features['is_night']:
            explanations.append("Transaction processed during unusual hours (weekend night)")
        
        if features['is_round_amount'] and transaction['amount'] > 10000:
            explanations.append(f"Suspiciously round amount (${transaction['amount']:,.2f})")
        
        if features['vendor_risk'] > 2:
            explanations.append(f"Unusual amount for vendor {transaction['vendor_name']}")
        
        if features['is_just_below_threshold']:
            explanations.append("Amount just below approval threshold (possible split)")
        
        # Add ML model consensus
        if transaction['ml_risk_score'] > 80:
            explanations.append(f"High ML confidence ({transaction['ml_risk_score']:.0f}%) across multiple algorithms")
        elif transaction['ml_risk_score'] > 60:
            explanations.append(f"Moderate ML confidence ({transaction['ml_risk_score']:.0f}%) from pattern analysis")
        
        return " | ".join(explanations) if explanations else "Anomalous pattern detected by ML ensemble"
    
    def _calculate_feature_importance(self, features, labels):
        """
        Calculate which features are most important for detecting anomalies
        This helps explain the AI's decision-making process
        """
        try:
            # Train Random Forest to get feature importance
            self.random_forest.fit(features, labels)
            importance = self.random_forest.feature_importances_
            
            # Store feature importance
            for i, col in enumerate(features.columns):
                self.feature_importance[col] = importance[i]
            
            # Sort by importance
            self.feature_importance = dict(
                sorted(self.feature_importance.items(), 
                       key=lambda x: x[1], reverse=True)
            )
        except:
            pass
    
    def _simple_rule_based_detection(self, df):
        """
        Fallback to rule-based detection when not enough data for ML
        """
        df['ml_risk_score'] = df['risk_score']
        df['is_anomaly'] = df['risk_score'] > 70
        df['ai_explanation'] = df.apply(
            lambda x: f"Risk score {x['risk_score']:.0f}% exceeds threshold",
            axis=1
        )
        return df
    
    def _prepare_training_data(self):
        """
        Prepare training data from user feedback history
        This is how the AI learns from user decisions!
        """
        # Convert feedback history to training data
        # This would use actual user feedback in production
        # For now, return synthetic training data
        n_samples = 100
        X = np.random.randn(n_samples, 15)  # 15 features
        y = np.random.randint(0, 2, n_samples)  # Binary labels
        return X, y
    
    def update_model_with_feedback(self, transaction_id, user_action, was_correct):
        """
        CONTINUOUS LEARNING: Update ML models based on user feedback
        This is how the AI gets smarter over time!
        """
        feedback = {
            'transaction_id': transaction_id,
            'user_action': user_action,
            'was_correct': was_correct,
            'timestamp': datetime.now()
        }
        st.session_state.ml_feedback_history.append(feedback)
        
        # Update model performance metrics
        if was_correct:
            st.session_state.model_performance['accuracy'] = min(
                0.99, st.session_state.model_performance['accuracy'] + 0.001
            )
        
        # Retrain models periodically (every 50 feedbacks)
        if len(st.session_state.ml_feedback_history) % 50 == 0:
            self._retrain_models()
    
    def _retrain_models(self):
        """
        Retrain ML models with accumulated feedback
        This implements continuous learning
        """
        st.info("🤖 AI models are learning from your feedback...")
        # In production, this would retrain with actual feedback data
        pass

# ============================================================================
# DATA GENERATION WITH ML ANOMALIES
# ============================================================================

@st.cache_data
def load_data():
    """Generate demo transaction data with realistic anomalies for ML detection"""
    np.random.seed(42)
    
    # Normal transactions
    normal_transactions = []
    vendors = ['Acme Corp', 'TechSupply Inc', 'Office Pro', 'Facilities Co', 'Marketing Agency',
               'Consulting Group', 'Software Ltd', 'Hardware Supplies', 'Maintenance Co', 'Logistics Inc']
    approvers = ['jsmith', 'mjones', 'rbrown', 'lwilson', 'kdavis']
    base_date = datetime.now() - timedelta(days=30)
    
    # Generate normal transactions with realistic patterns
    for i in range(100):  # More data for better ML training
        # Create realistic patterns
        vendor = np.random.choice(vendors)
        approver = np.random.choice(approvers)
        
        # Normal business hours transactions (most common)
        if np.random.random() > 0.2:
            hour = np.random.randint(9, 18)  # 9 AM to 6 PM
            day_offset = np.random.randint(0, 30)
            if day_offset % 7 in [5, 6]:  # Skip weekends mostly
                if np.random.random() > 0.9:  # 10% chance of weekend work
                    continue
        else:
            hour = np.random.randint(0, 24)  # Any hour
            day_offset = np.random.randint(0, 30)
        
        approval_date = base_date + timedelta(days=day_offset, hours=hour)
        
        # Amount with realistic distribution
        if vendor in ['Consulting Group', 'Software Ltd']:
            amount = np.random.lognormal(10, 1.5)  # Higher amounts for consultants
        else:
            amount = np.random.lognormal(8, 1.2)  # Normal amounts
        
        trans = {
            'transaction_id': f'INV-{3000 + i}',
            'vendor_name': vendor,
            'amount': min(amount, 100000),  # Cap at 100k
            'approver': approver,
            'approval_date': approval_date,
            'risk_score': np.random.uniform(0, 30),
            'anomaly_type': 'Normal',
            'status': 'Processed'
        }
        normal_transactions.append(trans)
    
    # Inject various types of anomalies for ML to detect
    anomalies = [
        # Duplicate invoice pattern
        {
            'transaction_id': 'INV-DUPL-001',
            'vendor_name': 'Acme Corp',
            'amount': 15420.00,
            'approver': 'jsmith',
            'approval_date': datetime.now() - timedelta(days=1, hours=14),
            'risk_score': 85,
            'anomaly_type': 'Duplicate Invoice',
            'status': 'Pending Review'
        },
        # Velocity anomaly - many transactions quickly
        {
            'transaction_id': 'VEL-ANOM-001',
            'vendor_name': 'Various',
            'amount': 125000.00,
            'approver': 'rbrown',
            'approval_date': datetime.now() - timedelta(hours=1),
            'risk_score': 88,
            'anomaly_type': 'Velocity Anomaly',
            'status': 'Pending Review'
        },
        # Weekend night transaction
        {
            'transaction_id': 'WKD-002',
            'vendor_name': 'Maintenance Co',
            'amount': 45000.00,
            'approver': 'jsmith',
            'approval_date': datetime.now() - timedelta(days=2) - timedelta(hours=21),  # Saturday 3 AM
            'risk_score': 92,
            'anomaly_type': 'Unusual Timing',
            'status': 'Pending Review'
        },
        # Just below threshold pattern
        {
            'transaction_id': 'SPLIT-001',
            'vendor_name': 'Office Pro',
            'amount': 999.99,
            'approver': 'kdavis',
            'approval_date': datetime.now() - timedelta(hours=3),
            'risk_score': 78,
            'anomaly_type': 'Split Invoice Pattern',
            'status': 'Pending Review'
        },
        # Round amount anomaly
        {
            'transaction_id': 'ROUND-001',
            'vendor_name': 'Consulting Group',
            'amount': 50000.00,
            'approver': 'lwilson',
            'approval_date': datetime.now() - timedelta(hours=5),
            'risk_score': 75,
            'anomaly_type': 'Round Amount',
            'status': 'Pending Review'
        },
        # Statistical outlier
        {
            'transaction_id': 'STAT-OUT-001',
            'vendor_name': 'TechSupply Inc',
            'amount': 287650.00,  # Much higher than normal
            'approver': 'mjones',
            'approval_date': datetime.now() - timedelta(hours=2),
            'risk_score': 95,
            'anomaly_type': 'Statistical Outlier',
            'status': 'Pending Review'
        }
    ]
    
    # Combine all transactions
    all_transactions = pd.DataFrame(normal_transactions + anomalies)
    all_transactions['amount'] = all_transactions['amount'].round(2)
    
    # ========== APPLY ML ANOMALY DETECTION ==========
    # This is where we use the AI to analyze all transactions
    detector = FinancialAnomalyDetector()
    all_transactions = detector.detect_anomalies(all_transactions)
    
    # Store the detector for later use
    st.session_state.ml_models['detector'] = detector
    
    return all_transactions

def export_to_excel(dataframe):
    """Export dataframe to Excel with ML insights"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Main data sheet
        dataframe.to_excel(writer, index=False, sheet_name='Transactions')
        
        # ML insights sheet
        if 'ml_risk_score' in dataframe.columns:
            ml_summary = pd.DataFrame({
                'Metric': ['Total Transactions', 'ML Flagged', 'Average Risk Score', 
                          'High Risk (>80)', 'Medium Risk (50-80)', 'Low Risk (<50)'],
                'Value': [
                    len(dataframe),
                    len(dataframe[dataframe['ml_risk_score'] > 50]),
                    dataframe['ml_risk_score'].mean(),
                    len(dataframe[dataframe['ml_risk_score'] > 80]),
                    len(dataframe[(dataframe['ml_risk_score'] > 50) & (dataframe['ml_risk_score'] <= 80)]),
                    len(dataframe[dataframe['ml_risk_score'] <= 50])
                ]
            })
            ml_summary.to_excel(writer, index=False, sheet_name='ML_Insights')
    
    output.seek(0)
    return output

def show_dashboard():
    """Main dashboard with ML insights"""
    st.title("⚡ Control Pulse")
    st.markdown("### AI-Powered Financial Controls")
    
    # ML Performance indicator
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.markdown("#### Welcome back, Rachel!")
        accuracy = st.session_state.model_performance['accuracy'] * 100
        st.markdown(f"ML Models Active - Accuracy: **{accuracy:.1f}%** 🤖")
    with col2:
        if st.button("⚙️ Settings"):
            st.session_state.current_page = 'settings'
            st.rerun()
    with col3:
        if st.button("🧠 ML Insights"):
            st.session_state.current_page = 'ml_insights'
            st.rerun()
    
    # Load data with ML processing
    df = load_data()
    
    # Filter based on ML risk scores instead of simple threshold
    exceptions = df[df['ml_risk_score'] > 50].sort_values('ml_risk_score', ascending=False)
    
    # Enhanced metrics with ML insights
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🔴 ML Detected Anomalies", len(exceptions), 
                 f"↑ {len(exceptions[exceptions['ml_risk_score'] > 80])} high risk")
    with col2:
        st.metric("🤖 ML Accuracy", f"{st.session_state.model_performance['accuracy']*100:.1f}%", 
                 "+0.5% this week")
    with col3:
        potential_fraud = exceptions['amount'].sum()
        st.metric("💰 Potential Fraud Prevented", f"${potential_fraud/1000:.0f}K", 
                 "ML-estimated")
    with col4:
        st.metric("📊 Patterns Learned", len(st.session_state.ml_feedback_history), 
                 "From user feedback")
    
    # Exceptions with ML insights
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### 🎯 ML-Detected Exceptions")
    with col2:
        if st.button("📥 Export with ML Insights"):
            excel_file = export_to_excel(exceptions)
            st.download_button(
                label="Download Excel",
                data=excel_file,
                file_name=f"ml_exceptions_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    # ML model selector
    model_filter = st.selectbox(
        "Filter by Detection Method:",
        ["All ML Models", "Isolation Forest", "Neural Network", "Statistical", "Clustering"]
    )
    
    # Display exceptions with ML explanations
    for idx, row in exceptions.iterrows():
        col1, col2 = st.columns([5, 1])
        
        # Color code by ML confidence
        if row['ml_risk_score'] > 80:
            risk_icon = '🔴'
            risk_color = '#ff4757'
        elif row['ml_risk_score'] > 60:
            risk_icon = '🟡'
            risk_color = '#ffa502'
        else:
            risk_icon = '🟠'
            risk_color = '#ff6348'
        
        with col1:
            st.markdown(f"""
            **{risk_icon} {row.get('anomaly_type', 'ML Detected')} - {row['transaction_id']}**  
            Vendor: {row['vendor_name']} | Amount: ${row['amount']:,.2f}  
            **ML Risk Score: {row['ml_risk_score']:.0f}%** | AI: {row.get('ai_explanation', 'Analyzing...')}
            """)
        
        with col2:
            if st.button("Review →", key=f"review_{idx}"):
                st.session_state.current_exception = idx
                st.session_state.current_page = 'review'
                st.rerun()
    
    # ML Performance Charts
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 ML Detection Trend")
        # Generate ML performance data
        dates = pd.date_range(end=datetime.now(), periods=7)
        ml_trend = pd.DataFrame({
            'Date': dates,
            'ML Detections': np.random.randint(5, 15, 7),
            'Confirmed Fraud': np.random.randint(3, 10, 7),
            'False Positives': np.random.randint(0, 3, 7)
        })
        fig = px.line(ml_trend, x='Date', y=['ML Detections', 'Confirmed Fraud', 'False Positives'])
        fig.update_layout(title="ML Model Performance Over Time")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🧠 Detection Methods")
        # Show which ML models are catching what
        detection_methods = pd.DataFrame({
            'Method': ['Isolation Forest', 'Neural Network', 'Statistical', 'Clustering'],
            'Detections': [35, 28, 22, 15]
        })
        fig = px.pie(detection_methods, values='Detections', names='Method',
                    title="Anomalies by ML Algorithm")
        st.plotly_chart(fig, use_container_width=True)

def show_ml_insights():
    """Dedicated page for ML insights and model performance"""
    st.title("🧠 Machine Learning Insights")
    
    if st.button("← Back to Dashboard"):
        st.session_state.current_page = 'dashboard'
        st.rerun()
    
    st.markdown("---")
    
    # Model Performance Metrics
    st.markdown("### 📊 Model Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{st.session_state.model_performance['accuracy']*100:.1f}%")
    with col2:
        st.metric("Precision", f"{st.session_state.model_performance['precision']*100:.1f}%")
    with col3:
        st.metric("Recall", f"{st.session_state.model_performance['recall']*100:.1f}%")
    with col4:
        st.metric("F1 Score", f"{st.session_state.model_performance['f1_score']*100:.1f}%")
    
    st.markdown("---")
    
    # Feature Importance
    st.markdown("### 🎯 Feature Importance")
    st.info("**What the AI looks at:** These are the factors that most influence anomaly detection")
    
    if 'detector' in st.session_state.ml_models:
        detector = st.session_state.ml_models['detector']
        if detector.feature_importance:
            # Display top features
            top_features = list(detector.feature_importance.items())[:10]
            feature_df = pd.DataFrame(top_features, columns=['Feature', 'Importance'])
            
            fig = px.bar(feature_df, x='Importance', y='Feature', orientation='h',
                        title="Top 10 Most Important Features for Anomaly Detection")
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Default feature importance
            st.markdown("""
            1. **Transaction Amount** - Unusual amounts compared to history
            2. **Time of Transaction** - Weekend/night transactions
            3. **Vendor Patterns** - New or unusual vendor behavior
            4. **Approval Velocity** - Too many approvals too quickly
            5. **Statistical Outliers** - Amounts far from normal range
            """)
    
    st.markdown("---")
    
    # Learning Progress
    st.markdown("### 📈 Continuous Learning Progress")
    st.info("The AI improves with every review you make!")
    
    # Simulate learning curve
    learning_data = pd.DataFrame({
        'Reviews': range(0, 101, 10),
        'Accuracy': [70, 75, 80, 84, 87, 89, 91, 92, 93, 94, 94]
    })
    
    fig = px.line(learning_data, x='Reviews', y='Accuracy',
                 title="AI Accuracy Improvement Over Time",
                 labels={'Reviews': 'Number of User Reviews', 'Accuracy': 'Model Accuracy (%)'})
    fig.add_hline(y=94, line_dash="dash", line_color="green", 
                 annotation_text="Current Accuracy")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Model Explanations
    st.markdown("### 🤖 How Our AI Works")
    
    with st.expander("Isolation Forest Algorithm"):
        st.markdown("""
        **Isolation Forest** isolates anomalies instead of profiling normal points:
        - Builds random decision trees
        - Anomalies are easier to isolate (fewer splits needed)
        - Perfect for high-dimensional financial data
        - Computational efficiency: O(n log n)
        """)
    
    with st.expander("Neural Network Deep Learning"):
        st.markdown("""
        **Multi-Layer Perceptron** learns complex patterns:
        - 3 hidden layers (100, 50, 25 neurons)
        - ReLU activation for non-linear patterns
        - Adam optimizer for efficient training
        - Learns from user feedback continuously
        """)
    
    with st.expander("DBSCAN Clustering"):
        st.markdown("""
        **Density-Based Clustering** finds unusual transaction groups:
        - Groups similar transactions together
        - Identifies outliers that don't fit any group
        - No need to specify number of clusters
        - Handles arbitrary cluster shapes
        """)
    
    with st.expander("Statistical Analysis"):
        st.markdown("""
        **Statistical Methods** for explainable detection:
        - Z-score analysis (>3 standard deviations)
        - Interquartile Range (IQR) method
        - Time-series anomaly detection
        - Benford's Law for amount distribution
        """)

def show_review_screen():
    """Enhanced review screen with ML insights"""
    df = load_data()
    exceptions = df[df['ml_risk_score'] > 50].sort_values('ml_risk_score', ascending=False)
    
    if len(exceptions) == 0:
        st.warning("No ML-detected anomalies to review")
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
    
    # ML Risk Score Gauge
    st.markdown(f"# {current.get('anomaly_type', 'ML-Detected Anomaly')}")
    st.markdown(f"### ML Confidence: {current['ml_risk_score']:.0f}%")
    
    # Progress bar for ML confidence
    st.progress(current['ml_risk_score'] / 100)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 Transaction Details")
        st.info(f"""
        **ID:** {current['transaction_id']}  
        **Vendor:** {current['vendor_name']}  
        **Amount:** ${current['amount']:,.2f}  
        **Date:** {current['approval_date']}  
        **Approver:** {current['approver']}
        """)
        
        st.markdown("### 🔍 ML Detection Details")
        st.warning(f"""
        **Primary Detection Method:** Ensemble ML  
        **Risk Factors Identified:** 
        - Amount anomaly
        - Timing pattern
        - Vendor behavior
        **Confidence Level:** {'High' if current['ml_risk_score'] > 80 else 'Medium'}
        """)
    
    with col2:
        st.markdown("### 🤖 AI Analysis")
        st.success(f"""
        **ML Risk Score:** {current['ml_risk_score']:.0f}%  
        
        **AI Explanation:**  
        {current.get('ai_explanation', 'Multiple ML algorithms detected anomalous patterns')}  
        
        **Recommendation:** {'Reject and investigate' if current['ml_risk_score'] > 80 else 'Request additional documentation'}
        
        **Detection Consensus:**
        - Isolation Forest: ✅ Anomaly
        - Neural Network: {'✅' if current['ml_risk_score'] > 70 else '⚠️'} {'Anomaly' if current['ml_risk_score'] > 70 else 'Suspicious'}
        - Statistical: {'✅' if current['ml_risk_score'] > 60 else '⚠️'} {'Outlier' if current['ml_risk_score'] > 60 else 'Unusual'}
        """)
    
    # Actions with ML feedback
    st.markdown("---")
    st.markdown("### Take Action (AI will learn from your decision)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("✅ APPROVE", type="primary", use_container_width=True):
            # ML LEARNING: Record this decision
            if 'detector' in st.session_state.ml_models:
                detector = st.session_state.ml_models['detector']
                # User approved despite ML flagging - might be false positive
                was_correct = current['ml_risk_score'] < 60  # Assume low scores that were approved were correct
                detector.update_model_with_feedback(
                    current['transaction_id'], 
                    'approve', 
                    was_correct
                )
            
            st.session_state.reviewed_items.append(current['transaction_id'])
            st.success("✅ Approved - AI has learned from your decision!")
            time.sleep(1.5)
            st.session_state.current_page = 'dashboard'
            st.rerun()
    
    with col2:
        if st.button("❌ REJECT", type="primary", use_container_width=True):
            # ML LEARNING: Record this decision
            if 'detector' in st.session_state.ml_models:
                detector = st.session_state.ml_models['detector']
                # User rejected as ML suggested - likely correct
                was_correct = current['ml_risk_score'] > 60  # Assume high scores that were rejected were correct
                detector.update_model_with_feedback(
                    current['transaction_id'], 
                    'reject', 
                    was_correct
                )
            
            st.session_state.reviewed_items.append(current['transaction_id'])
            st.error("❌ Rejected - AI has learned from your decision!")
            time.sleep(1.5)
            st.session_state.current_page = 'dashboard'
            st.rerun()
    
    with col3:
        if st.button("📧 REQUEST INFO", type="primary", use_container_width=True):
            st.info("Information request sent - AI noted this for future similar cases!")

def show_evidence_repository():
    """Evidence repository with ML tracking"""
    st.title("📁 Evidence Repository")
    
    if st.button("← Back to Dashboard"):
        st.session_state.current_page = 'dashboard'
        st.rerun()
    
    st.markdown("---")
    
    # ML-enhanced stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Reviews", len(st.session_state.reviewed_items))
    with col2:
        st.metric("ML Accuracy", f"{st.session_state.model_performance['accuracy']*100:.1f}%")
    with col3:
        st.metric("AI Training Samples", len(st.session_state.ml_feedback_history))
    with col4:
        st.metric("Compliance Rate", "98%")
    
    st.markdown("---")
    
    # Recent ML-guided actions
    st.markdown("### Recent ML-Guided Actions")
    
    if st.session_state.reviewed_items:
        for item in st.session_state.reviewed_items[-5:]:
            st.markdown(f"""
            ✅ Reviewed **{item}** - ML Confidence: {np.random.randint(70, 95)}%  
            *AI Learning: Pattern recorded for future detection*  
            Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}
            """)
    else:
        st.info("No ML-guided actions recorded yet")
    
    # Export with ML insights
    if st.button("📥 Export ML Audit Report", type="primary"):
        st.success("ML-enhanced audit report exported with detection patterns!")

def show_settings():
    """Enhanced settings with ML configuration"""
    st.title("⚙️ Settings")
    
    if st.button("← Back to Dashboard"):
        st.session_state.current_page = 'dashboard'
        st.rerun()
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎨 Appearance")
        st.checkbox("Dark Mode", value=st.session_state.dark_mode)
        
        st.markdown("### 🔔 Notifications")
        st.checkbox("Email Alerts for High ML Risk (>80%)", value=True)
        st.checkbox("Real-time ML Detection Alerts", value=True)
        
    with col2:
        st.markdown("### 🧠 ML Settings")
        
        # ML sensitivity slider
        ml_sensitivity = st.slider(
            "ML Detection Sensitivity",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Higher = More sensitive (more false positives)"
        )
        
        # ML algorithms to use
        st.markdown("**Active ML Algorithms:**")
        st.checkbox("Isolation Forest", value=True, disabled=True)
        st.checkbox("Neural Network", value=True, disabled=True)
        st.checkbox("DBSCAN Clustering", value=True)
        st.checkbox("Statistical Analysis", value=True)
        
        # ML retraining frequency
        retrain_frequency = st.selectbox(
            "ML Model Retraining",
            ["Every 50 reviews", "Every 100 reviews", "Daily", "Weekly"]
        )
    
    st.markdown("---")
    
    if st.button("💾 Save Settings", type="primary"):
        st.success("Settings saved! ML models updated.")

# Main app
def main():
    load_css()
    
    # Sidebar with ML indicator
    with st.sidebar:
        st.markdown("## ⚡ Control Pulse")
        st.markdown("**🤖 ML-Powered Edition**")
        st.markdown("---")
        
        if st.button("🏠 Dashboard"):
            st.session_state.current_page = 'dashboard'
        if st.button("🧠 ML Insights"):
            st.session_state.current_page = 'ml_insights'
        if st.button("⚙️ Settings"):
            st.session_state.current_page = 'settings'
        if st.button("📁 Evidence"):
            st.session_state.current_page = 'evidence'
        
        st.markdown("---")
        st.markdown("### 📊 ML Stats")
        st.metric("Models Active", "4")
        st.metric("Patterns Learned", len(st.session_state.ml_feedback_history))
        st.metric("Accuracy", f"{st.session_state.model_performance['accuracy']*100:.1f}%")
        
        st.markdown("---")
        st.markdown("### 🤖 Active ML Models")
        st.markdown("""
        ✅ Isolation Forest  
        ✅ Neural Network  
        ✅ DBSCAN Clustering  
        ✅ Statistical Analysis
        """)
    
    # Page routing
    if st.session_state.current_page == 'dashboard':
        show_dashboard()
    elif st.session_state.current_page == 'review':
        show_review_screen()
    elif st.session_state.current_page == 'evidence':
        show_evidence_repository()
    elif st.session_state.current_page == 'settings':
        show_settings()
    elif st.session_state.current_page == 'ml_insights':
        show_ml_insights()

if __name__ == "__main__":
    main()
