from pyspark.sql import SparkSession
from pyspark.sql.functions import length, mean, col, count, desc, explode
import pandas as pd
import plotly.express as px
import json
import os

# --- Configuration ---
INPUT_HDFS_PATH = "hdfs:///user/LLM_DataPrep/final/llm_dataset"
LOCAL_OUTPUT_DIR = "/home/project/raw_data/"
# --- End Configuration ---

spark = SparkSession.builder.appName("Step4_DataAnalysis").getOrCreate()

try:
    print("STEP 4: Starting Data Analysis and Visualization...")
    
    df = spark.read.parquet(INPUT_HDFS_PATH)
    
    # 1. Calculate Statistics
    length_df = df.withColumn("text_length", length(col("main_text")))
    stats = length_df.select(
        count(col("id")).alias("total_records"),
        mean(col("text_length")).alias("mean_text_length")
    ).collect()[0].asDict()
    
    # 2. Calculate Top Token Frequency
    token_counts = df.select(explode(col("tokens")).alias("token")) \
                     .groupBy("token") \
                     .count() \
                     .sort(desc("count")) \
                     .limit(20)

    # 3. Generate Visualization
    token_counts_pd = token_counts.toPandas()
    fig = px.bar(
        token_counts_pd,
        x='token',
        y='count',
        title='Top 20 Token Frequency in Biomedical Dataset'
    )
    
    # 4. Save Outputs
    local_viz_path = os.path.join(LOCAL_OUTPUT_DIR, "token_frequency_chart.html")
    fig.write_html(local_viz_path)

    analysis_results = {
        "Total Records": stats['total_records'],
        "Mean Text Length (Chars)": round(stats['mean_text_length'], 2),
        "Top 20 Tokens": token_counts_pd.to_dict('records'),
        "Visualization File": local_viz_path
    }
    
    print(f"Analysis results saved to {local_viz_path}")

except Exception as e:
    print(f"Error in Step 4: {e}")
    raise
finally:
    spark.stop()
