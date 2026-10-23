import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import BranchPythonOperator, PythonOperator

# ค่าเริ่มต้นที่ใช้ร่วมกันทุก task ใน DAG นี้
default_args = {
    "owner": "workshop2",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}
# บันทึกโมเดลฟุตบอล
MODEL_DIR = "/opt/airflow/models/football_models"
# ถ้าโมเดลมี R² ต่ำกว่านี้จะไม่ deploy
R2_THRESHOLD = 0.8  

# โหลดข้อมูลฟุตบอลจากไฟล์ CSV
def extract_data(**kwargs):
    """โหลดข้อมูลฟุตบอลจากไฟล์ CSV"""
    import pandas as pd

    file_path = "/opt/airflow/data/football_players.csv"
    df = pd.read_csv(file_path)

    print(f"โหลดข้อมูลฟุตบอลสำเร็จ: {len(df)} แถว")
    print(f"คอลัมน์ทั้งหมด: {df.columns.tolist()}")
    print(df.head().to_string())
    ti = kwargs["ti"]
    ti.xcom_push(key="data_path", value=file_path)

# เตรียมข้อมูลสำหรับการทำนายมูลค่านักเตะ
def prepare_data(**kwargs):
    """เตรียมข้อมูลนักเตะและแบ่งข้อมูลเป็น train/test set"""
    import pandas as pd
    from sklearn.model_selection import train_test_split

    ti = kwargs["ti"]
    file_path = ti.xcom_pull(
        task_ids="extract_data", key="data_path"
    )
    df = pd.read_csv(file_path)

    feature_columns = [
        "age",
        "overall_rating",
        "potential",
        "height_cm",
        "weight_kgs",
    ]
    target_column = "value_euro"
    name_columns = ["name", "full_name"]

    df = df[name_columns + feature_columns + [target_column]].copy()

    # เก็บชื่อไว้ค้นหาและแสดงผลตอน demo
    for column in name_columns:
        df[column] = (
            df[column]
            .fillna("")
            .astype(str)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    # ทำความสะอาดข้อมูลตัวเลข
    numeric_columns = feature_columns + [target_column]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df[numeric_columns] = df[numeric_columns].replace(
        [float("inf"), float("-inf")], float("nan")
    )
    df = df.dropna(subset=numeric_columns).reset_index(drop=True)

    if len(df) < 2:
        raise ValueError("ข้อมูลไม่เพียงพอสำหรับแบ่ง train/test")

    # ชื่อและ features อยู่ใน record เดียวกัน
    # เพื่อค้นหานักเตะแล้วหยิบข้อมูลของคนนั้นไปทำนาย
    players = df[
        name_columns + feature_columns
    ].to_dict(orient="records")

    # โมเดลใช้เฉพาะข้อมูลตัวเลข ไม่ใช้ชื่อ
    features = df[feature_columns].values.tolist()
    targets = df[target_column].tolist()

    # แบ่งข้อมูลตามแนวของอาจารย์
    X_train, X_test, y_train, y_test = train_test_split(
        features, targets, test_size=0.2, random_state=42
    )

    if len(y_test) < 2:
        raise ValueError("ข้อมูล test ต้องมีอย่างน้อย 2 แถวเพื่อคำนวณ R²")

    ti.xcom_push(key="X_train", value=X_train)
    ti.xcom_push(key="X_test", value=X_test)
    ti.xcom_push(key="y_train", value=y_train)
    ti.xcom_push(key="y_test", value=y_test)

    # เก็บข้อมูลสำหรับส่วน demo ที่จะทำภายหลัง
    ti.xcom_push(key="players", value=players)
    ti.xcom_push(key="feature_columns", value=feature_columns)

    print(
        f"แบ่งข้อมูลแล้ว: train {len(X_train)} แถว, "
        f"test {len(X_test)} แถว"
    )
    print(f"เก็บข้อมูลนักเตะสำหรับค้นหา: {len(players)} คน")

def train_model(**kwargs):
    """ฝึกโมเดลทำนายมูลค่านักเตะ แล้วบันทึกไฟล์แยกตามรอบรัน"""
    import joblib
    from sklearn.ensemble import RandomForestRegressor

    ti = kwargs["ti"]
    X_train = ti.xcom_pull(task_ids="prepare_data", key="X_train")
    y_train = ti.xcom_pull(task_ids="prepare_data", key="y_train")

    # ฝึกโมเดลด้วย features และมูลค่าจริงของนักเตะ
    model = RandomForestRegressor(n_estimators=50,random_state=42)
    model.fit(X_train, y_train)

    # บันทึกโมเดลตามแนวของอาจารย์
    os.makedirs(MODEL_DIR, exist_ok=True)

    run_id = "".join(char if char.isalnum() or char in "-_" else "_" for char in kwargs["run_id"])
    candidate_path = os.path.join(MODEL_DIR, f"candidate_{run_id}.pkl")
    joblib.dump(model, candidate_path)

    # ส่ง path ให้ task ประเมินผลนำไปโหลดต่อ
    ti.xcom_push(key="candidate_model_path",value=candidate_path,)
    print(f"ฝึกโมเดลด้วยข้อมูล {len(X_train)} แถว")
    print(f"เทรนโมเดลเสร็จ บันทึกไว้ที่: {candidate_path}")

def evaluate_model(**kwargs):
    """ประเมินโมเดลทำนายมูลค่านักเตะบน test set"""
    import joblib
    from sklearn.metrics import mean_absolute_error, r2_score

    ti = kwargs["ti"]
    candidate_path = ti.xcom_pull(task_ids="train_model", key="candidate_model_path")
    X_test = ti.xcom_pull(task_ids="prepare_data", key="X_test")
    y_test = ti.xcom_pull(task_ids="prepare_data", key="y_test")

    model = joblib.load(candidate_path)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    ti.xcom_push(key="mae", value=float(mae))
    ti.xcom_push(key="r2", value=float(r2))

    print(f"MAE: {mae:,.2f} ยูโร")
    print(f"R²: {r2:.4f}")

def decide_deploy(**kwargs):
    """เลือก deploy เมื่อ R² สูงกว่าเกณฑ์"""
    ti = kwargs["ti"]
    r2 = ti.xcom_pull(task_ids="evaluate_model", key="r2")

    if r2 > R2_THRESHOLD:
        print(f"R² {r2:.4f} > {R2_THRESHOLD} -> deploy")
        return "deploy_model"
    else:
        print(f"R² {r2:.4f} ไม่ผ่านเกณฑ์ -> skip")
        return "skip_deploy"

def deploy_model(**kwargs):
    """บันทึกโมเดลที่ผ่านเกณฑ์ พร้อมข้อมูลค้นหาสำหรับ demo"""
    import joblib
    import shutil

    ti = kwargs["ti"]
    candidate_path = ti.xcom_pull(task_ids="train_model", key="candidate_model_path")
    players = ti.xcom_pull(task_ids="prepare_data", key="players")
    feature_columns = ti.xcom_pull(task_ids="prepare_data", key="feature_columns")
    current_model_path = os.path.join(MODEL_DIR, "current_model.pkl")
    demo_data_path = os.path.join(MODEL_DIR, "demo_data.pkl")
    shutil.copyfile(candidate_path, current_model_path)

    # เก็บข้อมูลไว้ให้หน้า demo โหลดใช้งานภายหลัง
    joblib.dump(
        {
            "players": players,
            "feature_columns": feature_columns,
        },
        demo_data_path,
    )

    print(f"Deploy โมเดลสำเร็จ: {current_model_path}")
    print(f"บันทึกข้อมูลค้นหานักเตะ: {demo_data_path}")
    return "deployed"

def smoke_test(**kwargs):
    """โหลดโมเดลที่ deploy แล้ว ทดลองทำนายราคานักเตะ"""
    import joblib
    import math

    current_model_path = os.path.join(MODEL_DIR, "current_model.pkl")
    demo_data_path = os.path.join(MODEL_DIR, "demo_data.pkl")

    model = joblib.load(current_model_path)
    demo_data = joblib.load(demo_data_path)

    players = demo_data["players"]
    feature_columns = demo_data["feature_columns"]

    sample_players = players[:3]

    if not sample_players:
        raise ValueError("ไม่มีข้อมูลนักเตะสำหรับ smoke test")

    # เรียง features ให้ตรงกับตอน train
    sample_features = [
        [player[column] for column in feature_columns]
        for player in sample_players
    ]
    predictions = model.predict(sample_features)

    if len(predictions) != len(sample_players):
        raise ValueError("จำนวนผลทำนายไม่ตรงกับจำนวนตัวอย่าง")

    for player, predicted in zip(sample_players, predictions):
        if not math.isfinite(float(predicted)) or predicted < 0:
            raise ValueError(f"ราคาทำนายไม่ถูกต้อง: {predicted}")

        name = player["full_name"] or player["name"]
        print(f"{name}: ราคาทำนาย {predicted:,.2f} ยูโร")

    print("Smoke test ผ่าน: โหลดโมเดลและทำนายได้จริง")
    return "passed"

def skip_deploy(**kwargs):
    """ข้ามการ deploy เมื่อโมเดลไม่ผ่านเกณฑ์"""
    ti = kwargs["ti"]
    r2 = ti.xcom_pull(task_ids="evaluate_model", key="r2")

    print(f"ข้าม deploy: R² {r2:.4f} ไม่ผ่านเกณฑ์")
    return "skipped"

def log_result(**kwargs):
    """สรุปผลทั้งกรณี deploy และ skip"""
    ti = kwargs["ti"]

    mae = ti.xcom_pull(task_ids="evaluate_model", key="mae")
    r2 = ti.xcom_pull(task_ids="evaluate_model", key="r2")
    smoke_result = ti.xcom_pull(task_ids="smoke_test")
    skip_result = ti.xcom_pull(task_ids="skip_deploy")

    if smoke_result == "passed":
        outcome = "Deploy สำเร็จ และผ่าน smoke test"
    elif skip_result == "skipped":
        outcome = "ข้าม deploy เพราะไม่ผ่านเกณฑ์"
    else:
        raise ValueError("ไม่พบผลลัพธ์จากเส้นทาง deploy หรือ skip")

    print("===== สรุปผล Football Pipeline =====")
    print(f"MAE: {mae:,.2f} ยูโร")
    print(f"R²: {r2:.4f}")
    print(f"เกณฑ์ deploy: R² > {R2_THRESHOLD}")
    print(f"ผลลัพธ์: {outcome}")

# นิยาม DAG
with DAG(
    dag_id="football_pipeline_dag",
    default_args=default_args,
    description="ML pipeline ทำนายมูลค่านักเตะจากข้อมูล FIFA",
    schedule=None,
    start_date=datetime(2026, 8, 1),
    catchup=False,
    max_active_runs=1,
    tags=["workshop2", "football", "ml-pipeline"],
) as dag:

    extract_task = PythonOperator(task_id="extract_data",python_callable=extract_data,)
    prepare_task = PythonOperator(task_id="prepare_data",python_callable=prepare_data,)
    train_task = PythonOperator(task_id="train_model",python_callable=train_model,)
    evaluate_task = PythonOperator(task_id="evaluate_model",python_callable=evaluate_model,)

    decide_task = BranchPythonOperator(task_id="decide_deploy", python_callable=decide_deploy)
    
    deploy_task = PythonOperator(task_id="deploy_model",python_callable=deploy_model,)
    smoke_task = PythonOperator(task_id="smoke_test",python_callable=smoke_test,)
    skip_task = PythonOperator(task_id="skip_deploy",python_callable=skip_deploy,)
    log_task = PythonOperator(task_id="log_result",python_callable=log_result,trigger_rule="none_failed_or_skipped",)

    # ลำดับการรันของ task
    extract_task >> prepare_task >> train_task >> evaluate_task >> decide_task
    decide_task >> deploy_task >> smoke_task >> log_task
    decide_task >> skip_task >> log_task
