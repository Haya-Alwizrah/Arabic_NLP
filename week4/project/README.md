# Project 4 : Arabic RAG system with Gradio interface

---

## Installation:

```bash
pip install langchain-text-splitters groq chromadb gradio
```

## Data set:
used 30000 arabic wikipedia samples from:
[Jr23xd23/ArabicText-Large](https://huggingface.co/datasets/Jr23xd23/ArabicText-Large).

## Embedding Model:
Model used for embeddings:
[Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2](https://huggingface.co/Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2).

## chunk functions:
The project includes 3 chunking methods:
- **Fixed Chunking**: Splits text using fixed chunk sizes.
- **Recursive Chunking**: Splits text recursively using separators.
- **Semantic Chunking**: Splits text based on sentence similarity.

## LLM Model:
The project uses **llama-3.1-8b-instant** through **Groq API** to generate answers.

## Interface:
Simple UI built with Gradio:

<img width="2037" height="544" alt="image" src="https://github.com/user-attachments/assets/23bfa07c-1e3e-4910-8bc3-1807779a5c7d" />
