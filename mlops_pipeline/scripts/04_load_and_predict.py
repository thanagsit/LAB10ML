from sklearn.datasets import load_iris
import mlflow
import os
import pathlib

def load_and_predict():
    """
    Simulates a production scenario by loading a model via an alias
    in the MLflow Model Registry and using it for prediction.
    """
    MODEL_NAME = "iris-classifier-prod"
    MODEL_ALIAS = "Staging"  # เปลี่ยนเป็น "Production" ได้ถ้าตั้ง alias ไว้แล้ว

    # แนะนำ: ชี้ให้ใช้ local tracking store (กันพาธ C:\ บน Linux runners)
    local_uri = pathlib.Path("mlruns").resolve().as_uri()  # file:///.../mlruns
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", local_uri))

    print(f"Loading model '{MODEL_NAME}' with alias '{MODEL_ALIAS}'...")

    # โหลดโมเดลจาก Model Registry ด้วย alias (รูปแบบ models:/<name>@<alias>)
    try:
        model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
    except mlflow.exceptions.MlflowException as e:
        print(f"\nError loading model: {e}")
        print(f"Please make sure a model version has the '{MODEL_ALIAS}' alias in the MLflow UI.")
        return

    # เตรียมตัวอย่างข้อมูล (แถวแรกของ Iris)
    X, y = load_iris(return_X_y=True, as_frame=False)
    sample_data = X[0:1]   # 1 แถว
    actual_label = y[0]

    # พยากรณ์
    prediction = model.predict(sample_data)

    print("-" * 30)
    print(f"Sample Data Features:\n{sample_data[0]}")
    print(f"Actual Label: {actual_label}")
    print(f"Predicted Label: {prediction[0]}")
    print("-" * 30)

if __name__ == "__main__":
    load_and_predict()
