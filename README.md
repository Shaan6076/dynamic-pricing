# 🛒 AI for Market Trend Analysis – Dynamic Pricing & Sales Prediction

This project was developed as part of the **AI & ML course (IIT Ropar)**.  
The goal is to build an **AI system that analyzes market data and predicts product sales** based on pricing, category, brand, and other attributes.

---

## Project Overview
- **Problem Statement**: Market pricing is dynamic, influenced by demand, category, and seasonality. Retailers need AI-driven insights to adjust prices and forecast sales.
- **Objective**:  
  - Analyze retail product data  
  - Engineer meaningful features (time, profit, lag-based)  
  - Train ML models to predict **sales units**  
  - Visualize results & trends  
  - Deploy a simple **Streamlit app** for interactive predictions  

---

## Dataset
- Dataset contains **252,000 entries** of retail products.  
- **Features include**:  
  - Date (transaction date)  
  - Price & Cost  
  - Product Category, Brand, Collection, Style  
  - Gender, Price Tier  
  - Sales (target variable)

---

## Workflow
1. **Data Cleaning**
   - Removed missing values (84,000 null sales handled)
   - Dropped duplicates

2. **Exploratory Data Analysis (EDA)**
   - Correlation heatmap (price, cost, sales)
   - Category/brand-wise sales patterns

3. **Feature Engineering**
   - Date features → year, month, day, weekday, weekend flag  
   - Profit & margin ratio  
   - Lag features (7-day rolling averages)  
   - One-hot encoding for categorical variables  

4. **Model Training**
   - Models tested: Linear Regression, Random Forest  
   - Final model: **Random Forest Regressor**  
   - Metrics:  
     - MAE: ~19.25  
     - RMSE: ~26.45  
     - R²: **0.9976**

5. **Evaluation**
   - Scatter plot: Actual vs Predicted  
   - Bar chart: 20 random samples (easy comparison)

6. **Visualization Dashboard (Notebook)**
   - Predicted Sales per Product (bar chart)  
   - Price vs Predicted Sales (scatter)  
   - Category vs Predicted Sales (bar chart)  
   - Actual vs Predicted comparisons  

7. **Deployment**
   - Built a **Streamlit app (`app.py`)** with:
     - Single product prediction (sliders & dropdowns)  
     - Multiple product prediction via CSV upload  
     - Evaluation toggle: Scatter plot or Bar chart  

---

## How to Run

### Run Notebook
1. **Clone the repo**  
   ```bash
   git clone <your-repo-link>
   cd dynamic-pricing

2. **Create virtual environment & install      dependencies**
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

3. **Open Juypter Notebook**
   juypter noteboook

4. **Run dynamic_pricing.ipynb**

5. **Run Streamlit App**

   Ensure your trained model + features are    saved (.pkl files)

   Run the app

   streamlit run app.py


   Open in browser → http://localhost:8501


**Results**

The model achieved R² ≈ 0.998, showing high accuracy.

Predictions respond realistically to changes in product attributes (e.g., lower prices or holiday season → higher sales).

The Streamlit app provides an easy-to-use interface for real-time sales prediction.



**Project Structure**

dynamic-pricing/
│── app.py                # Streamlit web app
│── notebooks/            # Jupyter notebooks (EDA, training, evaluation)
│── models/               # Saved model + feature list + y_test/y_pred
│── data/                 # Raw & processed data
│── README.md             # Project documentation
│── requirements.txt      # Dependencies

Author

Name: [Shaanveer Singh Cheema]

Course: Minor in AI (IIT Ropar)

Focus: AI for Market Trend Analysis (Dynamic Pricing & Prediction)