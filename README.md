
# FoodWise: ML-Based Food Demand & Waste Intelligence Platform

This repository contains the deployment-ready files for the FoodWise Streamlit application.

## Deployment Information

### Local Development

To run the FoodWise application locally, follow these steps:

1.  **Clone the repository** (if you haven't already):
    ```bash
    git clone https://github.com/your-username/foodwise.git
    cd foodwise
    ```

2.  **Install dependencies**:
    Ensure you have Python 3.9+ installed. Then install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Streamlit app**:
    Navigate to the `deployment` directory (if not already there) and run the `app.py` file:
    ```bash
    streamlit run app.py
    ```
    This will open the application in your web browser.

### Permanent Deployment (Streamlit Community Cloud)

The FoodWise application is designed for permanent deployment using Streamlit Community Cloud.

1.  **Create a Streamlit Community Cloud account** (if you don't have one).
2.  **Connect your GitHub repository** containing these deployment files to Streamlit Community Cloud.
3.  **Deploy the `app.py`** from the root of this deployment folder.

### Repository Structure

The `deployment/` folder has the following structure, containing all necessary files to run the app:

```
deployment/
├── app.py                      # Main Streamlit application file
├── foodwise_core_functions.py  # Core ML prediction and analysis logic
├── requirements.txt            # Python dependencies for the application
├── models/                     # Directory for trained ML models
│   ├── xgboost_model.joblib    # The XGBoost demand prediction model
│   └── imputation_values.joblib# Dictionary of imputation values for missing features
└── data/
    └── Food demand.csv         # Raw historical food demand data
