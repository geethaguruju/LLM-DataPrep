import streamlit as st
import pandas as pd
import plotly.express as px
import os
import pyarrow.parquet as pq
from collections import Counter
import itertools

# --- PASTEL COLOR PALETTE (For Aesthetic Consistency) ---
PASTEL_PALETTE = ['#B3E0C6', '#ADD8E6', '#FFB3B3', '#DCDCDC', '#FFFACD', '#C0C0C0', '#90EE90'] 
# Light Mint, Light Blue, Light Red, Light Gray, Light Yellow, Silver, Light Green

# --- Configuration: Point to your LOCAL Parquet output directories ---
LOCAL_STAGE1_PATH = "./local_data/cleaned/stage1_data" 
LOCAL_FINAL_PATH = "./local_data/final/llm_dataset" 
LOCAL_RAW_PATH = "./local_data/raw/hf_pubmed_parquet/" 

# --- Data Limit Configuration (Crucial for stability) ---
# Maximum records to load into Streamlit/Pandas memory for visualization
MAX_ROWS_FOR_DASHBOARD = 50000 
SAMPLE_ROWS_FOR_TABLES = 500 

# =========================================================
# --- ACTUAL COUNTS FROM PIPELINE RUN ---
# =========================================================
initial_count = 2209839
stage1_count = 2209839 # All records passed initial cleaning/filtering based on log
final_count = 2023233 

# Calculate the dropped count (Initial - Final)
dropped_count = initial_count - final_count # 2209839 - 2112844 = 186606
# =========================================================


# --- Data Loading Functions ---

@st.cache_data
def load_data_sample(path, cols, limit):
    """Loads a limited sample from a multi-file Parquet directory."""
    try:
        if not os.path.exists(path):
            return pd.DataFrame()
            
        # Use ParquetDataset to handle multi-file Spark output directory
        dataset = pq.ParquetDataset(path)
        full_table = dataset.read(columns=cols) 

        # Slice the table to limit the rows (Optimization for memory)
        if full_table.num_rows > limit:
            table = full_table.slice(0, limit)
        else:
            table = full_table
        
        return table.to_pandas()
    
    except Exception as e:
        # st.error(f"Error loading data from {path}: {e}") 
        return pd.DataFrame() 

def get_pipeline_data():
    """Load samples from each pipeline stage."""
    
    raw_cols = ["id", "title", "content"] 
    raw_df = load_data_sample(LOCAL_RAW_PATH, raw_cols, SAMPLE_ROWS_FOR_TABLES)

    stage1_cols = ["id", "title", "main_text"] 
    stage1_df = load_data_sample(LOCAL_STAGE1_PATH, stage1_cols, SAMPLE_ROWS_FOR_TABLES)

    final_cols = ["id", "main_text", "tokens"] 
    final_df = load_data_sample(LOCAL_FINAL_PATH, final_cols, MAX_ROWS_FOR_DASHBOARD)
    
    return raw_df, stage1_df, final_df

# --- Custom Analysis Functions ---

def get_top_ngrams(tokens_list, n=2, top_k=10):
    """Calculates top N-grams (collocations) from a list of token lists."""
    if not tokens_list:
        return pd.DataFrame()
        
    ngrams = []
    # Use itertools to create n-gram tuples from each document's token list
    for doc_tokens in tokens_list:
        ngrams.extend(zip(*(doc_tokens[i:] for i in range(n))))

    # Filter out n-grams starting with common stop words (optional, but improves quality)
    stop_words = {'the', 'a', 'an', 'is', 'of', 'and', 'in', 'to', 'for', 'with', 'on'}
    ngrams_filtered = [
        " ".join(gram) for gram in ngrams 
        if gram[0] not in stop_words and all(word != '' for word in gram)
    ]

    counter = Counter(ngrams_filtered)
    top_ngrams_df = pd.DataFrame(counter.most_common(top_k), columns=['ngram', 'count'])
    return top_ngrams_df

# --- Streamlit Layout ---
st.set_page_config(layout="wide")
st.title(" LLM DataPrep Pipeline: Biomedical Dataset Analysis")
st.subheader("Showcasing Data Transformation and Quality Assurance")

raw_df, stage1_df, final_df = get_pipeline_data()

if final_df.empty:
    st.error("Cannot proceed: Final pipeline data could not be loaded. Ensure Steps 1-3 ran correctly and the path is valid.")
    st.stop() 


# =========================================================
# PRE-COMPUTATIONS FOR ANALYTICS
# =========================================================
if 'main_text' in final_df.columns:
    final_df['token_count'] = final_df['tokens'].apply(len)
    final_df['text_length'] = final_df['main_text'].str.len()


# =========================================================
# SECTION 1: Records Flow Visualization
# =========================================================
st.header("1. Document Count Flow and Reduction")
st.caption("Tracking record count through cleaning, filtering, and deduplication (Using Actual Pipeline Counts).")

df_flow = pd.DataFrame({
    'Stage': ['Initial Load', 'Stage 1: Cleaned & Filtered', 'Stage 2/3: Deduplicated & Final'],
    'Records': [initial_count, stage1_count, final_count]
})

fig_flow = px.bar(
    df_flow,
    x='Stage',
    y='Records',
    text='Records',
    title="Records Flow Across Pipeline Stages",
    labels={'Records': 'Total Document Count'},
    color='Stage',
    color_discrete_sequence=['#C0C0C0', PASTEL_PALETTE[0], PASTEL_PALETTE[1]]
)
fig_flow.update_traces(texttemplate='%{text:,}', textposition='outside')
fig_flow.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', xaxis={'categoryorder':'array', 'categoryarray':['Initial Load', 'Stage 1: Cleaned & Filtered', 'Stage 2/3: Deduplicated & Final']})
st.plotly_chart(fig_flow, use_container_width=True)

st.markdown("---")

# =========================================================
# SECTION 2: Fine-Tuning Related Analytics
# =========================================================
st.header("2. LLM Fine-Tuning Metrics")

if 'main_text' in final_df.columns:
    
    col_metric_1, col_metric_2, col_metric_3, col_metric_4 = st.columns(4)
    with col_metric_1:
        st.metric(label="Loaded Sample Size", value=f"{len(final_df):,} records")
    with col_metric_2:
        mean_length_token = final_df['token_count'].mean()
        st.metric(label="Mean Doc Token Count", value=f"{mean_length_token:.0f} tokens")
    with col_metric_3:
        mean_length_char = final_df['text_length'].mean()
        st.metric(label="Mean Char Length", value=f"{mean_length_char:.0f} characters")
    with col_metric_4:
        dedupe_rate = (dropped_count / initial_count * 100) if initial_count > 0 else 0
        st.metric(label="Total Data Reduction", value=f"{dedupe_rate:.2f}%")

    st.markdown("### Token & Vocabulary Analysis")
    col_tok_freq, col_ngrams = st.columns(2)

    with col_tok_freq:
        st.subheader("Top 20 Single Token Frequency")
        
        if 'tokens' in final_df.columns:
            all_tokens = list(itertools.chain.from_iterable(final_df['tokens'].tolist()))
            token_series = pd.Series(all_tokens)
            token_counts = token_series.value_counts().reset_index()
            token_counts.columns = ['token', 'count']
            token_counts_top = token_counts.head(20)

            fig_freq = px.bar(
                token_counts_top,
                x='token',
                y='count',
                title="Top 20 Tokens (Identifying Stopwords/Bias)",
                labels={'token': 'Token', 'count': 'Frequency'},
                color_discrete_sequence=[PASTEL_PALETTE[2]]
            )
            st.plotly_chart(fig_freq, use_container_width=True)
        else:
            st.warning("Token data missing in final dataset.")

    with col_ngrams:
        st.subheader("Top 10 Bigrams (Collocations)")
        
        if 'tokens' in final_df.columns:
            ngram_df = get_top_ngrams(final_df['tokens'].tolist(), n=2, top_k=10)
            
            if not ngram_df.empty:
                fig_ngrams = px.bar(
                    ngram_df,
                    x='count',
                    y='ngram',
                    orientation='h',
                    title="Top 10 Bigrams (Collocations)",
                    labels={'ngram': 'Bigram', 'count': 'Frequency'},
                    color_discrete_sequence=[PASTEL_PALETTE[3]]
                )
                fig_ngrams.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_ngrams, use_container_width=True)
            else:
                st.warning("Could not calculate N-grams.")


    st.markdown("### Document Length Analysis")
    col_length_hist, col_length_box = st.columns(2)

    with col_length_hist:
        st.subheader("Token Count Histogram")

        # Histogram for Token Count Distribution 
        fig_hist = px.histogram(
            final_df, 
            x="token_count", 
            nbins=50, 
            title="Distribution of Tokens Per Document (Log Scale)",
            log_y=True, 
            labels={'token_count': 'Token Count Per Document'},
            color_discrete_sequence=[PASTEL_PALETTE[0]]
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_length_box:
        st.subheader("Token Count Box Plot")

        # Box Plot for Token Count 
        fig_box = px.box(
            final_df, 
            y="token_count", 
            title="Statistical Summary: Quartiles and Outliers",
            labels={'token_count': 'Token Count Per Document'},
            color_discrete_sequence=[PASTEL_PALETTE[1]]
        )
        # Focus on a reasonable range (e.g., up to the 99th percentile)
        fig_box.update_yaxes(range=[0, final_df['token_count'].quantile(0.99) * 1.5]) 
        st.plotly_chart(fig_box, use_container_width=True)
        
    st.markdown("---")

# =========================================================
# SECTION 3: Data Transformation Samples
# =========================================================
st.header("3. Pipeline Transformation Flow")
st.caption(f"Comparing a small data sample ({SAMPLE_ROWS_FOR_TABLES} records) at each stage.")

tab1, tab2, tab3 = st.tabs(["Raw Data (Input)", "Stage 1: Cleaned & Filtered", "Final Data (LLM Ready)"])

with tab1:
    if not raw_df.empty:
        st.info("Raw data often contains multiple content fields and noisy data.")
        st.dataframe(raw_df.head(10), use_container_width=True)
    else:
        st.warning(f"Raw data not found at {LOCAL_RAW_PATH}.")

with tab2:
    if not stage1_df.empty:
        st.success("Cleaning removed citations, standardized content into 'main_text', and filtered short documents. (No records removed in this step, according to the log: 2,209,839 initial vs 2,209,839 cleaned)")
        st.dataframe(stage1_df.head(10), use_container_width=True)
    else:
        st.warning(f"Stage 1 data not found at {LOCAL_STAGE1_PATH}.")

with tab3:
    st.info("Final data is deduplicated, tokenized, and includes the 'tokens' array, resulting in 2,112,844 final records.")
    st.dataframe(final_df[['id', 'main_text', 'tokens']].head(10), use_container_width=True)

st.markdown("---")

# =========================================================
# SECTION 4: Deduplication Summary (Donut Chart)
# =========================================================
st.header("4. Deduplication and Cleaning Impact Summary")

col_pie, col_text = st.columns([1, 1])

# Data for Donut Chart
df_impact = pd.DataFrame({
    'Metric': ['Final LLM Data', 'Records Removed (Noise, Short, Duplicates)'],
    'Count': [final_count, dropped_count]
})

with col_pie:
    fig_impact = px.pie(
        df_impact, 
        values='Count', 
        names='Metric', 
        title=f"Initial Data Distribution (Total: {initial_count:,})",
        color='Metric',
        color_discrete_map={
            'Final LLM Data': PASTEL_PALETTE[1], 
            'Records Removed (Noise, Short, Duplicates)': PASTEL_PALETTE[2]
        },
        hole=0.5 # Creates the donut shape
    )
    fig_impact.update_traces(textinfo='percent+label')
    st.plotly_chart(fig_impact, use_container_width=True)

with col_text:
    st.subheader("Pipeline Quality Assurance Report")
    st.markdown(f"""
    - **Total Initial Records:** {initial_count:,}
    - **Total Records Removed:** {dropped_count:,}
    - **Total Data Reduction Rate:** **{dedupe_rate:.2f}%**
    
    The removed records account for filtering out short or noisy documents (Stage 1) and removing near-duplicate documents (Stage 2: LSH). This ensures high-quality, non-redundant data for effective LLM fine-tuning.
    """)