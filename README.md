# Cogito x Antire RAG workshop

This repository contains code and resources for the Cogito x Antire RAG workshop held in Trondheim.

## Setup

1. Clone the repository:
   ```bash
    git clone https://github.com/pegesund/trondheim_session.git
    cd trondheim_session
    ```
2. Setup environment variables:
   - Copy the example environment file and fill in your API keys:
     ```bash
     cp .env.example .env
     ```
   - Edit the `.env` file to add your Colivara and OpenRouter API keys
   - To find your Colivara API key, [Colivara](https://docs.colivara.com/getting-started/quickstart)
   - To find your OpenRouter API key, [OpenRouter](https://openrouter.ai/docs/api/api-reference/api-keys/create-keys)

## Usage

To run the RAG script, use the following command:
```bash
uv run colivara_rag.py
```