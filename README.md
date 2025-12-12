# BioLLM Data Preparation Pipeline (Apache Airflow & Dataproc)

## Project Overview

This project demonstrates a robust, scalable Big Data ETL pipeline designed to process a large biomedical abstract dataset for fine-tuning a Large Language Model (LLM). The pipeline uses **Apache Spark** running on **Google Cloud Dataproc** for distributed processing, with **Apache Airflow** providing external orchestration and dependency management.

## Objectives Covered

* **1 (Data Processing):** Data Ingestion, Cleaning, Near-Deduplication (LSH), and Tokenization using PySpark.
* **2 (Orchestration):** Airflow orchestrates the sequential execution of PySpark jobs via SSH.
* **3 (Monitoring):** Verified using YARN Application History on Dataproc (screenshots required).
* **4 (Analysis/Viz):** Interactive data quality dashboard using Streamlit/Plotly.


## Part 1: Final Setup and Execution

### 1. Prerequisite Check

Before launching the dashboard or pipeline, ensure the following is complete:

1.  **Environment Active:** The Python environment (`airflow_env`) is active.
    ```bash
    conda activate airflow_env
    ```
2.  **Airflow Running:** The necessary services are running locally.
    ```bash
    # Start the workflow engine and UI backend
    airflow scheduler &
    airflow api-server -p 8080 & 
    ```
3.  **Data Exists:** The final processed Parquet data is in the folder for Streamlit (e.g., `./temp_streamlit_data`).
4.  **Airflow Login:** We are logged into the Airflow UI (`http://localhost:8080`) using the generated Admin password.


### 2. Trigger the Orchestrated ETL Pipeline

Once connected, the pipeline can be launched:

1.  In the Airflow UI, find the **`llm_dataprep_ssh_pipeline`** DAG.
2.  Select **Trigger DAG** manually to run the dag.

The tasks will turn green as they successfully execute `spark-submit` commands on the remote cluster, demonstrating **Objective 6**. 


## Part 2: Data Analysis and Visualization

This section launches the interactive dashboard for demonstrating data quality and analysis.

### 1. Dashboard Dependencies

Ensure the necessary visualization tools are installed:

```bash
pip install streamlit pandas pyarrow plotly 
```
### 2. Launching Dashboard

The Streamlit data dashboard can be launched by using:
```bash
python3 data_dashboard.py 
```

## Part 3: Training & Inference for TinyLlama Bio SFT

### Follow the steps below to set up your environment, train the model, and run inference.

1. Setup & Installation
Open the python Notebook Bigdata.ipynb
pip install transformers datasets accelerate sentencepiece bitsandbytes

2. Your bio-sft folder should look like:
tinyllama-bio-sft/
 ├── config.json
 ├── tokenizer.json
 ├── tokenizer.model
 ├── special_tokens_map.json
 ├── pytorch_model.bin (or adapter_model.bin)
 └── generation_config.json

3. Training the LLM
Step 1 — Prepare Dataset
Use the parquet dataset that was run through the pipeline
Step 2 — Run the Training Notebook
Run the next code blocks of Bigdata.ipynb

4. Inside the notebook you will:
Load TinyLlama
Train with LoRA or full fine-tuning
Save the final model to: tinyllama-bio-sft/

5. At any point, you can save it manually:
trainer.save_model("tinyllama-bio-sft/")
tokenizer.save_pretrained("tinyllama-bio-sft/")

After training, the model folder will contain:
    pytorch_model.bin
    tokenizer.json
    tokenizer.model
    special_tokens_map.json
    tokenizer_config.json
    generation_config.json

These are ALL required for inference.

### Running Inference

Open your inference notebook:
'' Inferencing.ipynb''

```
Make sure your model path is correct:

from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "tinyllama-bio-sft/"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype="auto"
)
```

Step 1 — Build Prompt Template

Use the same template as training:
```
def build_prompt(instruction, context):
    return f"""### Instruction:
{instruction}

### Context:
{context}

### Response:
"""
```

Step 2 — Run Inference
```
prompt = build_prompt(
    "Summarize the following biomedical text.",
    "Breast cancer is a heterogeneous disease..."
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

output_ids = model.generate(
    **inputs,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
)

print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

 5. Example Output: 
Breast cancer exhibits genetic and environmental diversity...
