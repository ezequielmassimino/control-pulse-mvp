"""
control_pulse_config.py
Configuration and database connection module for Control Pulse MVP
"""

import os
import json
import pandas as pd
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import psycopg2
import pymongo
import sqlalchemy
from sqlalchemy import create_engine
import requests
from datetime import datetime, timedelta

# ==========================================
# CONFIGURATION DATACLASSES
# ==========================================
@dataclass
class DatabaseConfig:
    """Database connection configuration"""
    db_type: str  # 'postgresql', 'mysql', 'mongodb', 'sqlite', 'api'
    host: str
    port: int
    database: str
    username: str
    password: str
    ssl_enabled: bool = False
    connection_pool_size: int = 5

@dataclass
class APIConfig:
    """API endpoint configuration"""
    base_url: str
    api_key: str
    headers: Dict[str, str]
    timeout: int = 30
    retry_count: int = 3
    rate_limit_per_minute: int = 60

@dataclass
class ModelConfig:
    """ML model configuration"""
    isolation_forest_params: Dict = None
    neural_network_params: Dict = None
    dbscan_params: Dict = None
    ensemble_weights: Dict[str, float] = None
    anomaly_threshold: float = 70.0
    retrain_frequency: str = "daily"  # daily, weekly, monthly
    
    def __post_init__(self):
        if self.isolation_forest_params is None:
            self.isolation_forest_params = {
                'contamination': 0.1,
                'n_estimators': 100,
                'max_features': 1.0,
                'bootstrap': False
            }
        
        if self.neural_network_params is None:
            self.neural_network_params = {
                'hidden_layer_sizes': (64, 32, 16),
                'activation': 'relu',
                'learning_rate': 'adaptive',
                'max_iter': 500
            }
        
        if self.dbscan_params is None:
            self.dbscan_params = {
                'eps': 0.5,
                'min_samples': 5,
                'metric': 'euclidean',
                'algorithm': 'auto'
            }
        
        if self.ensemble_weights is None:
            self.ensemble_weights = {
                'isolation_forest': 0.35,
                'neural_network': 0.30,
                'dbscan': 0.20,
                'statistical': 0.15
            }

# ==========================================
# DATABASE CONNECTION MANAGER
# ==========================================
class DatabaseConnectionManager:
    """Manages database connections for different database types"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.connection = None
        self.engine = None
    
    def connect(self):
        """Establish database connection"""
        if self.config.db_type == 'postgresql':
            return self._connect_postgresql()
        elif self.config.db_type == 'mysql':
            return self._connect_mysql()
        elif self.config.db_type == 'mongodb':
            return self._connect_mongodb()
        elif self.config.db_type == 'sqlite':
            return self._connect_sqlite()
        else:
            raise ValueError(f"Unsupported database type: {self.config.db_type}")
    
    def _connect_postgresql(self):
        """Connect to PostgreSQL database"""
        connection_string = (
            f"postgresql://{self.config.username}:{self.config.password}"
            f"@{self.config.host}:{self.config.port}/{self.config.database}"
        )
        
        if self.config.ssl_enabled:
            connection_string += "?sslmode=require"
        
        self.engine = create_engine(
            connection_string,
            pool_size=self.config.connection_pool_size,
            pool_pre_ping=True
        )
        return self.engine
    
    def _connect_mysql(self):
        """Connect to MySQL database"""
        connection_string = (
            f"mysql+pymysql://{self.config.username}:{self.config.password}"
            f"@{self.config.host}:{self.config.port}/{self.config.database}"
        )
        
        self.engine = create_engine(
            connection_string,
            pool_size=self.config.connection_pool_size,
            pool_pre_ping=True
        )
        return self.engine
    
    def _connect_mongodb(self):
        """Connect to MongoDB"""
        if self.config.username and self.config.password:
            connection_string = (
                f"mongodb://{self.config.username}:{self.config.password}"
                f"@{self.config.host}:{self.config.port}/{self.config.database}"
            )
        else:
            connection_string = f"mongodb://{self.config.host}:{self.config.port}/"
        
        client = pymongo.MongoClient(connection_string)
        self.connection = client[self.config.database]
        return self.connection
    
    def _connect_sqlite(self):
        """Connect to SQLite database"""
        connection_string = f"sqlite:///{self.config.database}"
        self.engine = create_engine(connection_string)
        return self.engine
    
    def execute_query(self, query: str, params: Dict = None) -> pd.DataFrame:
        """Execute SQL query and return results as DataFrame"""
        if self.engine:
            return pd.read_sql_query(query, self.engine, params=params)
        else:
            raise ConnectionError("No database connection established")
    
    def fetch_transactions(self, 
                          table_name: str = "transactions",
                          date_from: datetime = None,
                          date_to: datetime = None,
                          limit: int = None) -> pd.DataFrame:
        """Fetch transactions from database"""
        if self.config.db_type == 'mongodb':
            return self._fetch_from_mongodb(table_name, date_from, date_to, limit)
        else:
            return self._fetch_from_sql(table_name, date_from, date_to, limit)
    
    def _fetch_from_sql(self, table_name: str, date_from: datetime, 
                       date_to: datetime, limit: int) -> pd.DataFrame:
        """Fetch from SQL database"""
        query = f"SELECT * FROM {table_name}"
        conditions = []
        
        if date_from:
            conditions.append(f"date >= '{date_from}'")
        if date_to:
            conditions.append(f"date <= '{date_to}'")
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        if limit:
            query += f" LIMIT {limit}"
        
        return self.execute_query(query)
    
    def _fetch_from_mongodb(self, collection_name: str, date_from: datetime,
                           date_to: datetime, limit: int) -> pd.DataFrame:
        """Fetch from MongoDB"""
        collection = self.connection[collection_name]
        
        filter_dict = {}
        if date_from or date_to:
            filter_dict['date'] = {}
            if date_from:
                filter_dict['date']['$gte'] = date_from
            if date_to:
                filter_dict['date']['$lte'] = date_to
        
        cursor = collection.find(filter_dict)
        if limit:
            cursor = cursor.limit(limit)
        
        return pd.DataFrame(list(cursor))
    
    def write_results(self, df: pd.DataFrame, table_name: str = "anomaly_results"):
        """Write results back to database"""
        if self.engine:
            df.to_sql(table_name, self.engine, if_exists='append', index=False)
        elif self.connection:  # MongoDB
            records = df.to_dict('records')
            self.connection[table_name].insert_many(records)
        else:
            raise ConnectionError("No database connection established")
    
    def close(self):
        """Close database connection"""
        if self.engine:
            self.engine.dispose()
        elif self.connection:
            self.connection.client.close()

# ==========================================
# API CONNECTION MANAGER
# ==========================================
class APIConnectionManager:
    """Manages API connections for external data sources"""
    
    def __init__(self, config: APIConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update(config.headers)
        if config.api_key:
            self.session.headers['Authorization'] = f"Bearer {config.api_key}"
    
    def fetch_transactions(self, 
                          endpoint: str = "/transactions",
                          params: Dict = None) -> pd.DataFrame:
        """Fetch transactions from API"""
        url = f"{self.config.base_url}{endpoint}"
        
        for attempt in range(self.config.retry_count):
            try:
                response = self.session.get(
                    url,
                    params=params,
                    timeout=self.config.timeout
                )
                response.raise_for_status()
                
                data = response.json()
                
                # Handle different response formats
                if isinstance(data, list):
                    return pd.DataFrame(data)
                elif isinstance(data, dict):
                    if 'data' in data:
                        return pd.DataFrame(data['data'])
                    elif 'transactions' in data:
                        return pd.DataFrame(data['transactions'])
                    else:
                        return pd.DataFrame([data])
                else:
                    raise ValueError(f"Unexpected response format: {type(data)}")
                    
            except requests.exceptions.RequestException as e:
                if attempt == self.config.retry_count - 1:
                    raise ConnectionError(f"Failed to fetch data from API: {e}")
                continue
    
    def push_results(self, 
                    results: Dict,
                    endpoint: str = "/anomalies") -> bool:
        """Push anomaly detection results to API"""
        url = f"{self.config.base_url}{endpoint}"
        
        try:
            response = self.session.post(
                url,
                json=results,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to push results to API: {e}")

# ==========================================
# DATA PIPELINE MANAGER
# ==========================================
class DataPipelineManager:
    """Manages the complete data pipeline from source to results"""
    
    def __init__(self, 
                 db_config: Optional[DatabaseConfig] = None,
                 api_config: Optional[APIConfig] = None,
                 model_config: Optional[ModelConfig] = None):
        self.db_manager = DatabaseConnectionManager(db_config) if db_config else None
        self.api_manager = APIConnectionManager(api_config) if api_config else None
        self.model_config = model_config or ModelConfig()
        
    def fetch_data(self, 
                  source: str = 'database',
                  **kwargs) -> pd.DataFrame:
        """Fetch data from configured source"""
        if source == 'database' and self.db_manager:
            self.db_manager.connect()
            return self.db_manager.fetch_transactions(**kwargs)
        elif source == 'api' and self.api_manager:
            return self.api_manager.fetch_transactions(**kwargs)
        elif source == 'csv':
            return pd.read_csv(kwargs.get('file_path'))
        elif source == 'excel':
            return pd.read_excel(kwargs.get('file_path'))
        else:
            raise ValueError(f"Invalid or unconfigured source: {source}")
    
    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names across different data sources"""
        column_mapping = {
            # Common variations of column names
            'transaction_id': ['trans_id', 'txn_id', 'id', 'transaction_number'],
            'amount': ['amt', 'value', 'transaction_amount', 'total'],
            'date': ['transaction_date', 'created_at', 'timestamp', 'datetime'],
            'vendor': ['vendor_name', 'supplier', 'merchant', 'payee'],
            'approver': ['approved_by', 'user', 'authorizer', 'reviewer'],
            'description': ['desc', 'details', 'notes', 'memo'],
            'category': ['type', 'classification', 'expense_type', 'account']
        }
        
        for standard_name, variations in column_mapping.items():
            for col in df.columns:
                if col.lower() in [v.lower() for v in variations]:
                    df.rename(columns={col: standard_name}, inplace=True)
                    break
        
        return df
    
    def validate_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
        """Validate data and return cleaned dataframe with validation warnings"""
        warnings = []
        
        # Check for required columns
        if 'amount' not in df.columns:
            raise ValueError("Required column 'amount' not found in data")
        
        # Remove rows with null amounts
        null_amounts = df['amount'].isnull().sum()
        if null_amounts > 0:
            warnings.append(f"Removed {null_amounts} rows with null amounts")
            df = df.dropna(subset=['amount'])
        
        # Convert amount to numeric
        try:
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        except:
            warnings.append("Some amount values could not be converted to numeric")
        
        # Parse dates if present
        if 'date' in df.columns:
            try:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                invalid_dates = df['date'].isnull().sum()
                if invalid_dates > 0:
                    warnings.append(f"{invalid_dates} rows have invalid dates")
            except:
                warnings.append("Date column could not be parsed")
        
        # Remove duplicates based on transaction_id if present
        if 'transaction_id' in df.columns:
            duplicates = df.duplicated(subset=['transaction_id']).sum()
            if duplicates > 0:
                warnings.append(f"Removed {duplicates} duplicate transactions")
                df = df.drop_duplicates(subset=['transaction_id'])
        
        return df, warnings
    
    def save_results(self, 
                    results: pd.DataFrame,
                    destination: str = 'database',
                    **kwargs) -> bool:
        """Save processed results to configured destination"""
        if destination == 'database' and self.db_manager:
            self.db_manager.write_results(results, **kwargs)
            return True
        elif destination == 'api' and self.api_manager:
            results_dict = {
                'timestamp': datetime.now().isoformat(),
                'anomalies_detected': len(results[results['is_anomaly'] == 1]),
                'results': results.to_dict('records')
            }
            return self.api_manager.push_results(results_dict, **kwargs)
        elif destination == 'csv':
            results.to_csv(kwargs.get('file_path', 'results.csv'), index=False)
            return True
        elif destination == 'excel':
            results.to_excel(kwargs.get('file_path', 'results.xlsx'), index=False)
            return True
        else:
            raise ValueError(f"Invalid or unconfigured destination: {destination}")

# ==========================================
# CONFIGURATION LOADER
# ==========================================
def load_config_from_file(config_path: str) -> Dict:
    """Load configuration from JSON or YAML file"""
    with open(config_path, 'r') as f:
        if config_path.endswith('.json'):
            return json.load(f)
        elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
            import yaml
            return yaml.safe_load(f)
        else:
            raise ValueError("Configuration file must be JSON or YAML")

def load_config_from_env() -> Dict:
    """Load configuration from environment variables"""
    config = {}
    
    # Database configuration
    if os.getenv('CP_DB_TYPE'):
        config['database'] = {
            'db_type': os.getenv('CP_DB_TYPE'),
            'host': os.getenv('CP_DB_HOST', 'localhost'),
            'port': int(os.getenv('CP_DB_PORT', 5432)),
            'database': os.getenv('CP_DB_NAME'),
            'username': os.getenv('CP_DB_USER'),
            'password': os.getenv('CP_DB_PASSWORD'),
            'ssl_enabled': os.getenv('CP_DB_SSL', 'false').lower() == 'true'
        }
    
    # API configuration
    if os.getenv('CP_API_URL'):
        config['api'] = {
            'base_url': os.getenv('CP_API_URL'),
            'api_key': os.getenv('CP_API_KEY'),
            'timeout': int(os.getenv('CP_API_TIMEOUT', 30))
        }
    
    # Model configuration
    if os.getenv('CP_ANOMALY_THRESHOLD'):
        config['model'] = {
            'anomaly_threshold': float(os.getenv('CP_ANOMALY_THRESHOLD', 70))
        }
    
    return config

# ==========================================
# EXAMPLE USAGE
# ==========================================
if __name__ == "__main__":
    # Example 1: PostgreSQL Configuration
    pg_config = DatabaseConfig(
        db_type='postgresql',
        host='localhost',
        port=5432,
        database='transactions_db',
        username='user',
        password='password'
    )
    
    # Example 2: API Configuration
    api_config = APIConfig(
        base_url='https://api.example.com/v1',
        api_key='your-api-key',
        headers={'Content-Type': 'application/json'}
    )
    
    # Example 3: Complete Pipeline
    pipeline = DataPipelineManager(
        db_config=pg_config,
        api_config=api_config
    )
    
    # Fetch data
    df = pipeline.fetch_data(source='database', limit=1000)
    
    # Standardize and validate
    df = pipeline.standardize_columns(df)
    df, warnings = pipeline.validate_data(df)
    
    print(f"Loaded {len(df)} transactions")
    if warnings:
        print("Warnings:", warnings)
