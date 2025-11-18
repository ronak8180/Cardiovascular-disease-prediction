import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect
import pickle
import json
import os
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')
from logger import setup_logger

app = Flask(__name__)

# Set up logger
logger = setup_logger()

# Global variables to store the trained models and preprocessor
models = {}
preprocessor = None
feature_columns = None
prediction_history = []  # simple in-memory history store
HISTORY_FILE = 'prediction_history.json'

def load_history_from_file():
    global prediction_history
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    prediction_history = data
    except Exception as e:
        print(f"Warning: failed to load history file: {e}")

def save_history_to_file():
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(prediction_history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Warning: failed to save history file: {e}")

def load_trained_model():
    """Load the pre-trained model with 95% accuracy"""
    global models, preprocessor, feature_columns
    
    try:
        print("Loading pre-trained model...")
        
        # Check if model files exist
        if not os.path.exists('trained_model.pkl'):
            print("❌ Pre-trained model not found!")
            print("Please run 'python save_model.py' first to create the model files.")
            return False
        
        # Load the pre-trained model
        with open('trained_model.pkl', 'rb') as f:
            models['random_forest'] = pickle.load(f)
        
        # Load the preprocessor
        with open('preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
        
        # Load feature columns
        with open('feature_columns.pkl', 'rb') as f:
            feature_columns = pickle.load(f)
        
    
        return True
        
    except Exception as e:
        print(f"❌ Error loading pre-trained model: {str(e)}")
        print("Falling back to training a new model...")
        return train_new_model()

def train_new_model():
    """Fallback: Train a new model if pre-trained model fails to load"""
    global models, preprocessor, feature_columns
    
    print("Training new model as fallback...")
    # Load the dataset
    heart = pd.read_csv("processed_data.csv")
    
    # Separate target
    y = heart["heart_disease"]   # target column
    X = heart.drop(columns=["heart_disease"])   # features only
    
    # Store feature columns for later use
    feature_columns = X.columns.tolist()
    
    # Identify feature types
    num_features = X.select_dtypes(exclude="object").columns
    cat_features = X.select_dtypes(include="object").columns
    
    # Transformers
    numeric_transformer = StandardScaler()
    oh_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    
    # ColumnTransformer (only applied on features, not target!)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_features),
            ("cat", oh_transformer, cat_features),
        ]
    )
    
    # Apply SMOTE for data balancing
    print("Applying SMOTE for data balancing...")
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    
    # Split the data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
    )
    
    # Train Random Forest
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10)
    
    # Fit preprocessor and model
    X_train_processed = preprocessor.fit_transform(X_train)
    rf_model.fit(X_train_processed, y_train)
    
    # Store the trained model
    models['random_forest'] = rf_model
    
    # Evaluate the model
    X_test_processed = preprocessor.transform(X_test)
    y_pred = rf_model.predict(X_test_processed)
    y_pred_proba = rf_model.predict_proba(X_test_processed)
    
    # Calculate metrics
    accuracy = rf_model.score(X_test_processed, y_test)
    roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])

    
    print("Models trained successfully!")
    return True

def predict_heart_disease(input_data, model_name='random_forest'):
    """Make prediction using the trained model"""
    global models, preprocessor

    try:
        # Check if required components are loaded
        if feature_columns is None:
            print("Error: feature_columns not loaded")
            return None
        if preprocessor is None:
            print("Error: preprocessor not loaded")
            return None
        if model_name not in models or models[model_name] is None:
            print(f"Error: model '{model_name}' not loaded")
            return None

        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])

        # Ensure all required columns are present
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0  # Default value for missing columns

        # Reorder columns to match training data
        input_df = input_df[feature_columns]

        # Transform the input data
        input_processed = preprocessor.transform(input_df)

        # Make prediction
        model = models[model_name]
        prediction = model.predict(input_processed)[0]
        probabilities = model.predict_proba(input_processed)[0]

        # Calculate risk level
        prob_disease = probabilities[1]
        if prob_disease >= 0.7:
            risk_level = "High"
        elif prob_disease >= 0.4:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        return {
            'prediction': int(prediction),
            'probability_disease': float(prob_disease),
            'probability_no_disease': float(probabilities[0]),
            'risk_level': risk_level,
            'model_used': model_name
        }

    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return None

# Routes
@app.route('/')
def home():
    """Redirect to predict"""
    logger.info(f"Request received: {request.method} {request.path}")
    return redirect('/predict')



@app.route('/predict')
def predict_page():
    """Render the prediction page"""
    logger.info(f"Request received: {request.method} {request.path}")
    return render_template('index.html')

@app.route('/history')
def history():
    """Render the history page"""
    logger.info(f"Request received: {request.method} {request.path}")
    return render_template('history.html')

@app.route('/about')
def about():
    """Render the about page"""
    logger.info(f"Request received: {request.method} {request.path}")
    return render_template('about.html')

@app.route('/api/history')
def api_history():
    """Return prediction history (latest first)"""
    logger.info(f"Request received: {request.method} {request.path}")
    return jsonify({'items': prediction_history})

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    logger.info(f"Request received: {request.method} {request.path}")
    try:
        # Get form data
        form_data = request.form.to_dict()
        
        # Convert form data to the format expected by the model
        input_data = {
            'sex': form_data.get('sex', ''),
            'age_category': form_data.get('age_category', ''),
            'bmi': float(form_data.get('bmi', 0)),
            'bmi_group': form_data.get('bmi_group', ''),
            'general_health': form_data.get('general_health', ''),
            'checkup': form_data.get('checkup', ''),
            'exercise': form_data.get('exercise', ''),
            'diabetes': form_data.get('diabetes', ''),
            'depression': form_data.get('depression', ''),
            'arthritis': form_data.get('arthritis', ''),
            'skin_cancer': form_data.get('skin_cancer', ''),
            'other_cancer': form_data.get('other_cancer', ''),
            'smoking_history': form_data.get('smoking_history', ''),
            'alcohol_consumption': int(form_data.get('alcohol_consumption', 0)),
            'fruit_consumption': int(form_data.get('fruit_consumption', 0)),
            'vegetables_consumption': int(form_data.get('vegetables_consumption', 0)),
            'potato_consumption': int(form_data.get('potato_consumption', 0))
        }
        
        # Make prediction
        result = predict_heart_disease(input_data)
        
        if result:
            # Append to history (latest first)
            record = {
                'date': pd.Timestamp.now().strftime('%Y-%m-%d'),
                'time': pd.Timestamp.now().strftime('%H:%M'),
                'risk': result['risk_level'],
                'probability': round(result['probability_disease'] * 100, 1),
                'status': 'danger' if result['risk_level'] == 'High' else 'warning' if result['risk_level'] == 'Medium' else 'success',
                'factors': {
                    'age': input_data.get('age_category', ''),
                    'bmi': input_data.get('bmi_group', ''),
                    'exercise': input_data.get('exercise', ''),
                    'smoking': input_data.get('smoking_history', ''),
                    'diabetes': input_data.get('diabetes', '')
                }
            }
            prediction_history.insert(0, record)
            save_history_to_file()
            return jsonify(result)
        else:
            return jsonify({'error': 'Prediction failed'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    logger.info(f"Request received: {request.method} {request.path}")
    try:
        data = request.get_json()
        
        # Make prediction
        result = predict_heart_disease(data)
        
        if result:
            # Append to history (latest first)
            record = {
                'date': pd.Timestamp.now().strftime('%Y-%m-%d'),
                'time': pd.Timestamp.now().strftime('%H:%M'),
                'risk': result['risk_level'],
                'probability': round(result['probability_disease'] * 100, 1),
                'status': 'danger' if result['risk_level'] == 'High' else 'warning' if result['risk_level'] == 'Medium' else 'success',
                'factors': {
                    'age': data.get('age_category', ''),
                    'bmi': data.get('bmi_group', ''),
                    'exercise': data.get('exercise', ''),
                    'smoking': data.get('smoking_history', ''),
                    'diabetes': data.get('diabetes', '')
                }
            }
            prediction_history.insert(0, record)
            save_history_to_file()
            return jsonify(result)
        else:
            return jsonify({'error': 'Prediction failed'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Heart Disease Prediction App...")
    
    # Load the pre-trained model instead of training
    if load_trained_model():
        # Load existing history from file
        load_history_from_file()
        print("Starting Flask server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load model. Exiting...")
