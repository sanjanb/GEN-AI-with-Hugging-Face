# HuggingFace & LangChain Integration Demo

**Interactive notebook demonstrating HuggingFace model integration with LangChain for NLP tasks**  
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1O70N0zDCEIirsjaWoD3bUZtiXzgn2DNP?usp=sharing)

---

## Key Features
- **Framework Integration**: Combines HuggingFace transformers with LangChain pipelines 
- **Dual Model Usage**: Demonstrates both `HuggingFaceEndpoint` (API-based) and `HuggingFacePipeline` (local model) approaches
- **GPU Support**: Includes configuration for GPU acceleration using `accelerate` and `bitsandbytes` 
- **Modular Design**: Separates prompt templates, model loading, and execution logic

---

## Requirements
```bash
# Core libraries
pip install langchain-huggingface huggingface_hub transformers accelerate bitsandbytes langchain

# For GPU support (CUDA 11.8+ recommended)
pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Setup
1. **HuggingFace Token**  
   - Create a HuggingFace account and [generate a token](https://huggingface.co/settings/tokens)
   - Store token in Colab secrets (under `Runtime > Manage Secrets`) with key `HUGGINGFACEHUB_TOKEN` 

2. **Environment Configuration**  
   ```python
   from google.colab import userdata
   os.environ["HUGGINGFACEHUB_API_TOKEN"] = userdata.get("HUGGINGFACEHUB_TOKEN")
   ```

---

## Usage

### **1. HuggingFace Endpoint (API)**
```python
from langchain_huggingface import HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    max_length=128,
    temperature=0.7
)

# Example query
llm_chain = LLMChain(
    prompt=PromptTemplate(template="Question: {question}\nAnswer:", 
                         input_variables=["question"]),
    llm=llm
)
print(llm_chain.invoke("What is co-attention mechanism?"))
```

### **2. Local Pipeline (GPU)**
```python
from langchain_huggingface import HuggingFacePipeline

gpu_pipeline = HuggingFacePipeline.from_model_id(
    model_id="gpt2",
    task="text-generation",
    device=0,  # Use GPU
    pipeline_kwargs={"max_new_tokens": 100}
)

# Example query
chain = (PromptTemplate(...) | gpu_pipeline)
chain.invoke({"question": "Explain transformer architecture"})
```

---

## Troubleshooting
- **CUDA Errors**: Ensure GPU runtime is enabled in Colab (`Runtime > Change runtime type`) 
- **Model Access**: Some models require accepting license agreements on HuggingFace
- **Dependency Conflicts**: Use `!pip install --force-reinstall` for conflicting packages 

---

## Acknowledgements
- HuggingFace [transformers](https://github.com/huggingface/transformers) and [langchain-huggingface](https://github.com/huggingface/langchain-huggingface) integrations 
- GPU acceleration enabled by [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
```

**Key Enhancements:**  
1. Explicit GPU setup instructions with CUDA version note   
2. Clear separation of API vs local model workflows  
3. Direct links to required libraries' documentation   
4. Troubleshooting section addressing common Colab issues  

Would you like to add any specific sections or modify the technical depth?
