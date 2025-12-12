from airflow import DAG
from airflow.providers.ssh.operators.ssh import SSHOperator
from datetime import datetime, timedelta

# Define shared Spark configuration arguments
SPARK_ARGS = (
    "--master yarn "
    "--deploy-mode client "
    "--executor-memory 8g "
    "--num-executors 16 "
)
# This is the remote path on the Dataproc master node where your scripts live
PROJECT_DIR = '/home/project/code/' 

with DAG(
    dag_id='llm_dataprep_ssh_pipeline',
    start_date=datetime(2025, 12, 1),
    schedule_interval=timedelta(days=7),
    catchup=False,
    default_args={
        'owner': 'NYU Big Data Team',
        'retries': 1,
        'retry_delay': timedelta(minutes=5),
        'ssh_conn_id': 'dataprochost' # CRITICAL: References the connection created in Step 2
    },
    description='Orchestrates the modular LLM data preparation pipeline via SSH to Dataproc.'
) as dag:
    
    # Task 1: Data Cleaning and Staging (01_clean_stage.py)
    clean_data_task = SSHOperator(
        task_id='1_data_cleaning_and_staging',
        command=f"spark-submit {SPARK_ARGS} {PROJECT_DIR}01_clean_stage.py",
        # Ensures the job runs on the cluster
    )

    # Task 2: Deduplication (02_dedupe_lsh.py)
    dedupe_lsh_task = SSHOperator(
        task_id='2_near_deduplication_lsh',
        command=f"spark-submit {SPARK_ARGS} {PROJECT_DIR}02_dedupe_lsh.py",
    )
    
    # Task 3: Tokenization and Final Save (03_tokenize_final.py)
    tokenize_final_task = SSHOperator(
        task_id='3_tokenization_and_final_save',
        command=f"spark-submit {SPARK_ARGS} {PROJECT_DIR}03_tokenize_final.py",
    )

    # Task 4: Data Analysis and Visualization (04_analyze_data.py)
    analyze_data_task = SSHOperator(
        task_id='4_data_analysis_and_stats',
        command=f"spark-submit {SPARK_ARGS} {PROJECT_DIR}04_analyze_data.py",
    )

    # Define the workflow dependencies (ensuring sequential execution)
    clean_data_task >> dedupe_lsh_task >> tokenize_final_task >> analyze_data_task
