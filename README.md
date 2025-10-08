# House Price Prediction API

A machine learning powered API built with **FastAPI** to predict house prices.  
The model is trained using **Linear Regression (scikit-learn)** on a housing dataset with features such as area, bedrooms, bathrooms, stories, and amenities.

---

## Features
- **FastAPI** backend with auto-generated Swagger docs
- Preprocessing for categorical features (yes/no encoding, furnishing status one-hot)
- `/predict` endpoint → predicts house price from input JSON
- `/health` endpoint → simple health check
- Jupyter notebook included for training, EDA, and model comparison

---

## Installation

Clone the repo and create a virtual environment:

```bash
git clone https://github.com/<your-username>/house-price-prediction-api.git
cd house-price-prediction-api
python -m venv env
.\env\Scripts\activate   # Windows
source env/bin/activate  # Mac/Linux
pip install -r requirements.txt
```
----

## Run the API

Start the server with:

uvicorn house_api:app --reload


API will be available at:

Swagger UI → http://127.0.0.1:8000/docs

Health check → http://127.0.0.1:8000/health

## Screenshots

## 📊 Exploratory Data Analysis
Correlation Heatmap:
![Correlation Heatmap](assets/correlation_heatmap.png)

## 🚀 API in Action
Swagger UI:
![Swagger UI](assets/swagger_ui.png)

Example Prediction:
![Prediction Example](assets/prediction_example.png)

