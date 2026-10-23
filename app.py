import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import os

# 1. Page Configuration
st.set_page_config(page_title="Drink Calorie Analyzer", page_icon="🥤", layout="wide")

st.title("🥤 Highest Calorie Drinks Analyzer & Predictor")
st.markdown("This app uses a **Decision Tree Classifier** to analyze and predict drink categories based on nutritional values.")

# 2. Rebuilding DataFrame with safe list structures to avoid parsing errors
names = [
    "Baskin-Robbins Chocolate Oreo Shake (Large)",
    "Sonic Peanut Butter Fudge Shake (Large)",
    "Smoothie King The Hulk Strawberry (40 oz)",
    "Dairy Queen Chocolate Malt (Large)",
    "Dunkin Frozen Chocolate (Large Whole Milk)",
    "Jamba Juice Peanut Butter Mood Smoothie (Large)",
    "McDonalds Chocolate Shake (Large)",
    "Starbucks White Hot Chocolate (Venti Whole Milk)",
    "Panera Bread Frozen Caramel Latte (16 oz)",
    "Coca-Cola (Big Gulp 32 oz)"
]

calories = [2600, 2340, 1928, 1350, 1120, 1090, 840, 640, 590, 360]
sugar = [263, 196, 250, 170, 166, 169, 122, 78, 82, 98]
fat = [135, 140, 64, 45, 40, 29, 22, 24, 15, 0]
categories = ["Shake", "Shake", "Smoothie", "Shake", "Frozen Drink", "Smoothie", "Shake", "Hot Coffee/Chocolate", "Frozen Coffee", "Soda"]

df = pd.DataFrame({
    "Drink_Name": names,
    "Calories": calories,
    "Sugar_g": sugar,
    "Fat_g": fat,
    "Category": categories
})

# 3. Sidebar: Model Prediction Setup
st.sidebar.header("🤖 Predict Drink Category")

# Model ko load karna
model_path = "decision_tree_drink_model.pkl"
if os.path.exists(model_path):
    model = joblib.load(model_path)
    
    st.sidebar.write("Adjust nutritional values to predict the category:")
    # User Inputs via Sliders
    input_calories = st.sidebar.slider("Calories (kcal)", min_value=100, max_value=3000, value=1200, step=50)
    input_sugar = st.sidebar.slider("Sugar (grams)", min_value=0, max_value=300, value=130, step=5)
    input_fat = st.sidebar.slider("Fat (grams)", min_value=0, max_value=200, value=40, step=5)
    
    # Predict Button
    if st.sidebar.button("Predict Category"):
        user_data = pd.DataFrame([[input_calories, input_sugar, input_fat]], columns=['Calories', 'Sugar_g', 'Fat_g'])
        prediction = model.predict(user_data)
        
        st.sidebar.success(f"🎯 Predicted Category: **{prediction[0]}**")
else:
    st.sidebar.error(f"⚠️ `{model_path}` file not found! Please upload the trained model file to your GitHub repository.")

# 4. Main Page View: Dataset Display
st.subheader("📊 Top 10 Highest Calorie Drinks Dataset")
st.dataframe(df, use_container_width=True)

# 5. Main Page View: 3D Visualization
st.subheader("📦 3D Data Visualization (Calories vs Sugar vs Fat)")
fig = px.scatter_3d(
    df, 
    x='Calories', 
    y='Sugar_g', 
    z='Fat_g',
    color='Category',
    hover_name='Drink_Name',
    opacity=0.8,
    height=600
)
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
st.plotly_chart(fig, use_container_width=True)
