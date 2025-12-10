from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace, length, trim, when

# --- Configuration ---
RAW_PARQUET_PATH = "hdfs:///user/gg3039_nyu_edu/LLM_DataPrep/raw/hf_pubmed_parquet/" 
CLEANED_HDFS_PATH = "hdfs:///user/gg3039_nyu_edu/LLM_DataPrep/cleaned/stage1_data"
# --- End Configuration ---

spark = SparkSession.builder.appName("Step1_Clean_Stage").getOrCreate()

try:
    print(f"STEP 1: Starting Data Cleaning and Staging...")
    
    df = spark.read.parquet(RAW_PARQUET_PATH) 
    df = df.select("id", "title", "content", "contents", col("PMID").cast("string").alias("PMID"))
    
    initial_count = df.count()
    print(f"   Initial record count: {initial_count}")
    
    # Create main_text by combining/prioritizing content fields
    df_combined = df.withColumn(
        "main_text",
        when(col("content").isNotNull() & (length(col("content")) > 50), col("content"))
        .otherwise(col("contents"))
    ).drop("content", "contents")
    
    # Apply Cleaning: citations, multiple spaces, 'Abstract', and length filtering
    df_filtered = df_combined.filter(col("main_text").isNotNull() & (length(trim(col("main_text"))) > 50))
    df_cleaned = df_filtered.withColumn("main_text", regexp_replace(col("main_text"), r'\[\s*\d+(?:-\d+)?\s*\]', ' '))
    df_cleaned = df_cleaned.withColumn("main_text", regexp_replace(col("main_text"), r'\s{2,}', ' '))
    df_cleaned = df_cleaned.withColumn("main_text", regexp_replace(col("main_text"), r'Abstract', ''))
    df_cleaned = df_cleaned.withColumn("main_text", trim(col("main_text")))

    final_count = df_cleaned.count()
    print(f"   Cleaned records saved: {final_count}")
    
    df_cleaned.write.mode("overwrite").parquet(CLEANED_HDFS_PATH)
    print(f"   Stage 1 data written to {CLEANED_HDFS_PATH}")

except Exception as e:
    print(f"Error in Step 1: {e}")
    raise
finally:
    spark.stop()
