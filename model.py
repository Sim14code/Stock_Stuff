

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def load_and_preprocess_data(file_path):
    """Load and preprocess stock market data."""
    try:
        data = pd.read_csv(file_path)
        print("📌 Columns in dataset:", data.columns)
        
        # Check for required columns
        required_columns = {'open', 'high', 'low', 'close', 'volume'}
        missing_columns = required_columns - set(data.columns)
        if missing_columns:
            raise KeyError(f"❌ Missing columns in data.csv: {missing_columns}")
        
        # Feature Engineering
        data['price_change'] = (data['close'] - data['open']) / data['open'] * 100
        data['target'] = (data['close'].shift(-1) > data['close']).astype(int)
        data.dropna(inplace=True)
        
        return data
    except Exception as e:
        print(f"⚠ Error loading data: {e}")
        return None

def train_and_save_model(data, model_path='stock_market_model.pkl'):
    """Train a RandomForest model and save it."""
    features = ['open', 'high', 'low', 'close', 'volume', 'price_change']
    X = data[features]
    y = data['target']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Model Accuracy: {accuracy:.2f}")
    
    # Save the model
    joblib.dump(model, model_path)
    print(f"📦 Model saved as {model_path}")
    return model

if __name__ == "__main__":
    dataset_path = 'data.csv'  # Ensure this file is available
    model_save_path = 'stock_market_model.pkl'
    
    stock_data = load_and_preprocess_data(dataset_path)
    if stock_data is not None:
        train_and_save_model(stock_data, model_save_path)

