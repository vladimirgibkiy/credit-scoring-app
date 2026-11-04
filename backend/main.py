from fastapi import FastAPI
from pydantic import BaseModel
from catboost import CatBoostClassifier, Pool
import uvicorn
import psycopg2
import mlflow
import os
import time

app = FastAPI(title='Credit Scoring API', description='Сервис для оценки вероятности дефолта заемщика')

model = CatBoostClassifier()
model.load_model('credit_model.cbm')

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Credit_Scoring_Production")

def get_db_connection():
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "postgres"),
        database=os.getenv("DB_NAME", "scoring_db"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "postgres_password")
    )
    return conn

def init_db():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS scoring_logs (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            age INT,
            debt_ratio FLOAT,
            probability FLOAT,
            risk_category VARCHAR(10)
        );
    """)
    conn.commit()
    cur.close()
    conn.close()

@app.on_event("startup")
def startup_event():
    time.sleep(5)
    init_db()

class BorrowerData(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float
    NumberOfTimes90DaysLate: int
    age: int
    NumberOfTime30_59DaysPastDueNotWorse: int
    DebtRatio: float
    NumberOfOpenCreditLinesAndLoans: int

@app.get('/')
def read_root():
    return {'message': 'Сервер кредитного скоринга запущен и готов к работе'}

@app.post('/predict')
def predict_default(data: BorrowerData):
    try:
        input_data = [[
            data.RevolvingUtilizationOfUnsecuredLines,
            data.NumberOfTimes90DaysLate,
            data.age,
            data.NumberOfTime30_59DaysPastDueNotWorse,
            data.DebtRatio,
            data.NumberOfOpenCreditLinesAndLoans
        ]]
        
        probability = round(float(model.predict_proba(input_data)[0][1]), 4)
        
        if probability > 0.5:
            risk_category = 'Высокий'
        else:
            risk_category = 'Низкий'
            
        data_pool = Pool(data=input_data)
        shap_results = model.get_feature_importance(data=data_pool, type="ShapValues")
        shap_values = shap_results.flatten().tolist()
        
        shap_explanation = {}
        shap_explanation["RevolvingUtilizationOfUnsecuredLines"] = round(float(shap_values[0]), 4)
        shap_explanation["NumberOfTimes90DaysLate"] = round(float(shap_values[1]), 4)
        shap_explanation["age"] = round(float(shap_values[2]), 4)
        shap_explanation["NumberOfTime30_59DaysPastDueNotWorse"] = round(float(shap_values[3]), 4)
        shap_explanation["DebtRatio"] = round(float(shap_values[4]), 4)
        shap_explanation["NumberOfOpenCreditLinesAndLoans"] = round(float(shap_values[5]), 4)
        
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO scoring_logs (age, debt_ratio, probability, risk_category) VALUES (%s, %s, %s, %s)",
            (data.age, data.DebtRatio, probability, risk_category)
        )
        conn.commit()
        cur.close()
        conn.close()

        with mlflow.start_run(run_name="Inference_Request"):
            mlflow.log_param("borrower_age", data.age)
            mlflow.log_param("debt_ratio", data.DebtRatio)
            mlflow.log_metric("predicted_probability", probability)
        
        return {
            'probability': probability,
            'risk_category': risk_category,
            'shap_features': shap_explanation
        }

    except Exception as error:
        print(f"Ошибка в работе эндпоинта predict: {error}")
        return {
            'probability': 0.0,
            'risk_category': 'Ошибка',
            'shap_features': {}
        }

@app.get('/global_shap')
def get_global_shap():
    try:
        global_importance = model.get_feature_importance(type="PredictionValuesChange")
        weights = global_importance.flatten().tolist()
        
        return {
            "RevolvingUtilizationOfUnsecuredLines": round(float(weights[0]), 4),
            "NumberOfTimes90DaysLate": round(float(weights[1]), 4),
            "age": round(float(weights[2]), 4),
            "NumberOfTime30_59DaysPastDueNotWorse": round(float(weights[3]), 4),
            "DebtRatio": round(float(weights[4]), 4),
            "NumberOfOpenCreditLinesAndLoans": round(float(weights[5]), 4)
        }
    except Exception as error:
        print(f"Ошибка в работе эндпоинта global_shap: {error}")
        return {
            "RevolvingUtilizationOfUnsecuredLines": 0.0,
            "NumberOfTimes90DaysLate": 0.0,
            "age": 0.0,
            "NumberOfTime30_59DaysPastDueNotWorse": 0.0,
            "DebtRatio": 0.0,
            "NumberOfOpenCreditLinesAndLoans": 0.0
        }

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
