import subprocess
import sys
import time
import os

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Base command arguments for spark-submit
SPARK_SUBMIT = "spark-submit"
SPARK_ARGS = (
    "--master yarn "
    "--deploy-mode client "
    "--executor-memory 8g "
    "--num-executors 16 "
)

# List of pipeline steps in dependency order
PIPELINE_STEPS = [
    {
        "name": "STEP 1: Data Cleaning & Staging",
        "script": "01_clean_stage.py",
        "dependency": "None"
    },
    {
        "name": "STEP 2: Near-Deduplication (LSH)",
        "script": "02_dedupe_lsh.py",
        "dependency": "STEP 1"
    },
    {
        "name": "STEP 3: Tokenization & Final Save",
        "script": "03_tokenize_final.py",
        "dependency": "STEP 2"
    },
    {
        "name": "STEP 4: Data Analysis & Visualization",
        "script": "04_analyze_data.py",
        "dependency": "STEP 3"
    },
]
# --- End Configuration ---

def run_spark_job(step_name, script_file):
    """Executes a single spark-submit command."""
    full_command = f"{SPARK_SUBMIT} {SPARK_ARGS} {SCRIPT_DIR}/{script_file}"
    
    print(f"\n=======================================================")
    print(f"  Starting {step_name}")
    print(f"  Command: {full_command}")
    print(f"=======================================================")
    
    start_time = time.time()
    
    try:
        # Execute the command and capture output/errors
        result = subprocess.run(full_command, shell=True, check=True, 
                                capture_output=True, text=True)
        
        end_time = time.time()
        
        print(f"\n--- Output of {step_name} ---")
        print(result.stdout)
        
        print(f"{step_name} SUCCESSFUL in {round(end_time - start_time, 2)} seconds.")
        return True
        
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        print(f"\n {step_name} FAILED after {round(end_time - start_time, 2)} seconds.")
        print(f"Dependency Failed: Execution of {step_name} did not complete.")
        print("--- Execution Error ---")
        print(e.stderr)
        return False

def main():
    """Main orchestration function."""
    print("--- Starting LLM DataPrep Modular Pipeline ---")
    
    for step in PIPELINE_STEPS:
        success = run_spark_job(step["name"], step["script"])
        
        if not success:
            print("\n PIPELINE FAILURE! Halting execution due to failed dependency. 🚨")
            sys.exit(1)
            
    print("\n\n ALL PIPELINE STEPS COMPLETED SUCCESSFULLY! ")

if __name__ == "__main__":
    main()
