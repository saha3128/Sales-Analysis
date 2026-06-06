import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Sales Prediction API")

# Load models and encoders from pickle files
def load_model(filename):
    try:
        with open(f'models/{filename}.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        raise Exception(f"Model file not found: models/{filename}.pkl. Please run the notebook's model-saving cell first.")

try:
    best_model = load_model('best_model')
    scaler = load_model('scaler')
    le_category = load_model('le_category')
    le_product = load_model('le_product')
    le_city = load_model('le_city')
    feature_cols = load_model('feature_cols')
except Exception as e:
    print(f"❌ Error loading models: {str(e)}")
    print("📌 Execute the notebook's model-saving cell (cell 21) to create pickle files.")
    raise

class SalesOrder(BaseModel):
    quantity: int
    unit_price: float
    product_category: str
    product_name: str
    customer_city: str
    day: int
    day_of_week: int

@app.get("/")
def root():
    return {"message": "Sales Prediction API is running. Use POST /predict to make predictions."}

@app.post("/predict")
def predict(order: SalesOrder):
    """Predict Total Sales for a given order"""
    
    try:
        # Create DataFrame
        df = pd.DataFrame([{
            'Quantity': order.quantity,
            'Unit_Price': order.unit_price,
            'Product_Category': order.product_category,
            'Product_Name': order.product_name,
            'Customer_City': order.customer_city,
            'Day': order.day,
            'DayOfWeek': order.day_of_week,
        }])
        
        # Encode categorical variables
        df['Category_Encoded'] = le_category.transform([order.product_category])[0]
        df['Product_Encoded'] = le_product.transform([order.product_name])[0]
        df['City_Encoded'] = le_city.transform([order.customer_city])[0]
        
        # Create engineered features
        df['Is_Weekend'] = 1 if order.day_of_week >= 5 else 0
        df['Price_Per_Quantity'] = order.unit_price / order.quantity
        
        # These would need aggregated data from training set
        # For now, using default values
        df['Revenue_Per_City'] = 1125.0  # avg from training
        df['Revenue_Per_Category'] = 1375.0  # avg from training
        df['Revenue_Per_Product'] = 1500.0  # avg from training
        
        # Select features in correct order
        X = df[feature_cols]
        
        # Make prediction
        predicted_sales = best_model.predict(X)[0]
        
        return {
            "order": order.dict(),
            "predicted_total_sales": round(float(predicted_sales), 2),
            "status": "success"
        }
    
    except Exception as e:
        return {
            "error": str(e),
            "status": "failed"
        }