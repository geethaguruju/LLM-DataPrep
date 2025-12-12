## Training & Inference for TinyLlama Bio SFT

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
