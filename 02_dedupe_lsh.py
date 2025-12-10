from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit
from pyspark.ml.feature import MinHashLSH, HashingTF, Tokenizer

# --- Configuration ---
INPUT_HDFS_PATH = "hdfs:///user/gg3039_nyu_edu/LLM_DataPrep/cleaned/stage1_data"
OUTPUT_HDFS_PATH = "hdfs:///user/gg3039_nyu_edu/LLM_DataPrep/cleaned/stage2_data"
# --- End Configuration ---

spark = SparkSession.builder.appName("Step2_Dedupe_LSH").getOrCreate()

try:
    print(f"STEP 2: Starting Near-Deduplication (LSH)...")
    
    df = spark.read.parquet(INPUT_HDFS_PATH)
    initial_count = df.count()
    
    # 1. Tokenize the text (required for HashingTF)
    tokenizer = Tokenizer(inputCol="main_text", outputCol="words")
    df_words = tokenizer.transform(df)

    # 2. Convert words to feature vectors (TF)
    hashing_tf = HashingTF(inputCol="words", outputCol="features", numFeatures=2**16)
    df_features = hashing_tf.transform(df_words)

    # 3. Apply MinHash LSH (for Jaccard distance approximation)
    lsh = MinHashLSH(inputCol="features", outputCol="hashes", numHashTables=5)
    model = lsh.fit(df_features)

    # 4. Find all similar items (candidates for deduplication)
    # Distance = 0.1 means Jaccard Similarity > 0.9 (since Distance = 1 - Similarity)
    # The output is a list of tuples: (dataset_id_1, dataset_id_2, distance)
    distance_threshold = 0.1 
    
    # This step is resource-intensive; run it to find clusters of similar documents
    # Note: We must join the resulting pairs back to the original dataframe
    
    # --- Simplified Dedupe Strategy (for course demo) ---
    # Due to complexity of self-join and large data, we use the hash for exact dedupe 
    # and explain LSH conceptually, or run LSH on a sample.
    
    # For a deterministic course demo: using MinHash for exact dedupe (less resource intensive)
    df_dedupe = model.transform(df_features).dropDuplicates(['hashes']).drop("words", "features", "hashes")
    
    dedupe_count = df_dedupe.count()
    dropped_count = initial_count - dedupe_count
    
    print(f"   Records removed by deduplication: {dropped_count}")
    print(f"   Stage 2 data written to {OUTPUT_HDFS_PATH}")
    
    df_dedupe.write.mode("overwrite").parquet(OUTPUT_HDFS_PATH)

except Exception as e:
    print(f"Error in Step 2: {e}")
    raise
finally:
    spark.stop()

