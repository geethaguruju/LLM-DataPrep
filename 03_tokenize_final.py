from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer
from pyspark.sql.functions import col

# --- Configuration ---
INPUT_HDFS_PATH = "hdfs:///user/gg3039_nyu_edu/LLM_DataPrep/cleaned/stage2_data" # Input from dedupe
OUTPUT_HDFS_PATH = "hdfs:///user/gg3039_nyu_edu/LLM_DataPrep/final/llm_dataset" # Final Output
# --- End Configuration ---

spark = SparkSession.builder.appName("Step3_Tokenize_Final").getOrCreate()

try:
    print(f"STEP 3: Starting Tokenization and Final Save...")
    
    df = spark.read.parquet(INPUT_HDFS_PATH)
    
    # 1. Apply PySpark's basic tokenizer (demonstrating distributed tokenization)
    tokenizer_text = Tokenizer(inputCol="main_text", outputCol="tokens")
    df_tokenized = tokenizer_text.transform(df)

    tokenizer_title = Tokenizer(inputCol="title", outputCol="title_tokens")
    df_tokenized = tokenizer_title.transform(df_tokenized)
    
    # 2. Select final columns for LLM fine-tuning
    df_final = df_tokenized.select(
        col("id"), 
        col("PMID"), 
        col("title"),
        col("main_text"),    # Required for LLM fine-tuning input
        col("tokens"),      # Tokenized array
    )

    final_count = df_final.count()
    print(f"   Final records saved: {final_count}")

    # 3. Save the final dataset
    df_final.write.mode("overwrite").parquet(OUTPUT_HDFS_PATH)
    print(f"   Final LLM dataset written to {OUTPUT_HDFS_PATH}")

except Exception as e:
    print(f"Error in Step 3: {e}")
    raise
finally:
    spark.stop()

