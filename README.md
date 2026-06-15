---
license: apache-2.0
language:
- en
pipeline_tag: text-generation
tags:
- mistral
- fine-tuned
- lora
- unsloth
base_model: unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit
---

# Hephaestus PRISM: A LLM for alloy phase diagram prediction

### About the Name
This project has been updated from its original title to **Hephaestus PRISM**. 

* **Hephaestus:** Named after the ancient Greek god of blacksmiths, metalworking, metallurgy, and fire. He represents the ultimate mastery over forging elemental materials into complex structures, mirroring this AI's purpose in predicting complex alloy microstructures.
* **PRISM:** A structural acronym standing for **P**hase **R**elationship **I**nference & **S**ynthesis **M**odel. Just as a physical prism splits raw light into its component spectral colors, this model takes raw composition data and maps it across distinct thermodynamic phase boundaries.

---

You can chat with it or interactively draw generated phase diagrams directly in your browser—no local GPU required.

<video controls autoplay muted loop width="100%">
  <source src="https://cdn-uploads.huggingface.co/production/uploads/65096d0e623330a3a51cf6aa/dhs4kI7brJ8mDtY_wamFf.qt" type="video/mp4">
  Your browser does not support the video tag.
</video>

![image/png](https://cdn-uploads.huggingface.co/production/uploads/65096d0e623330a3a51cf6aa/zHPBs16jPVdkaeZCBCR1b.png)

Data used to train the model are [here](https://huggingface.co/datasets/Playingyoyo/aLLoyM-dataset).

## Model Details

- **Model Name**: Hephaestus PRISM
- **Base Model**: unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Training Framework**: Unsloth

## Usage

### Basic Usage

```python
from unsloth import FastLanguageModel
import torch
from huggingface_hub import login

# Authenticate with Hugging Face
login('YOUR_HF_TOKEN')

# Model configuration
max_seq_length = 2048
dtype = torch.bfloat16
load_in_4bit = True

print("Loading model from Hugging Face...")
# Load model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name='Playingyoyo/Hephaestus-PRISM',
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
FastLanguageModel.for_inference(model)
print("Model loaded successfully!")

# Define the question
question = "What phases form when Arsenic (40%) + Platinum (60%) are mixed at 400 K?" # Replace here with your own question

# Create prompt template
prompt = f"""### Instruction:
You are an expert in phase diagrams, thermodynamics, and materials science, specializing in binary alloy systems.

### Input:
{question}

### Output:
"""

# Tokenize input
inputs = tokenizer(
    [prompt],
    return_tensors='pt',
    truncation=True
).to('cuda')

# Generate response
print(f"\nGenerating response for: '{question}'")
with torch.no_grad():
    outputs = model.generate(
        **inputs, 
        max_new_tokens=512,
        use_cache=True,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

# Decode and extract the generated response
full_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

# Extract only the generated part after "### Output:"
if "### Output:" in full_output:
    generated_response = full_output.split("### Output:")[1].strip()
else:
    generated_response = full_output.strip()

print(f"\nAnswer:")
print("=" * 50)
print(generated_response)
print("=" * 50)

```

### Question Samples

Hephaestus PRISM was trained using a standardized prompt template for
consistency, which may make it sensitive to variations in prompt formulation. Users should
be aware that rephrasing questions or changing the input format may affect prediction qual-
ity. We encourage the community to experiment with different prompting approaches and
share effective strategies.

![Screenshot 2025-09-29 at 17.13.26](https://cdn-uploads.huggingface.co/production/uploads/65096d0e623330a3a51cf6aa/jS71z0wjdd_7feq2s2LN3.png)

## Training Configuration

- **Learning Rate**: 2e-4
- **Batch Size**: 16 (per device)
- **Gradient Accumulation Steps**: 4
- **LoRA Rank**: 16
- **LoRA Alpha**: 16
- **Target Modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
## License
Apache 2.0
## Citation
If you use this model, please cite:
```bibtex
@misc{Hephaestus_PRISM,
  title=,
  author=,
  year=,
  publisher=,
  howpublished=
}
```
