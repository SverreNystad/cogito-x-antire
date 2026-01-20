#!/usr/bin/env python3
"""
Visual RAG using ColiVara API - ColPali visual embeddings as a service.
No local GPU needed.
"""
import os
import sys
import argparse
from pathlib import Path

from dotenv import load_dotenv
from colivara_py import ColiVara
import requests

load_dotenv()

COLIVARA_API_KEY = os.getenv("COLI_AP_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE = "https://openrouter.ai/api/v1"


def get_client():
    if not COLIVARA_API_KEY:
        print("Error: COLI_AP_KEY not found in .env")
        sys.exit(1)
    return ColiVara(api_key=COLIVARA_API_KEY)


def index_documents(pdf_dir: Path, collection_name: str = "trondheim_docs"):
    """Upload and index PDFs with ColiVara."""
    client = get_client()

    # Create or get collection
    try:
        collection = client.create_collection(name=collection_name)
        print(f"Created collection: {collection_name}")
    except Exception as e:
        if "409" in str(e) or "already" in str(e).lower() or "conflict" in str(e).lower():
            print(f"Using existing collection: {collection_name}")
        else:
            raise e

    # Upload PDFs
    pdf_files = list(pdf_dir.glob("*.pdf"))
    print(f"Indexing {len(pdf_files)} PDFs...")

    for pdf_path in pdf_files:
        print(f"  Uploading: {pdf_path.name}...", end=" ", flush=True)
        try:
            doc = client.upsert_document(
                name=pdf_path.stem,
                collection_name=collection_name,
                document_path=pdf_path,
                metadata={"filename": pdf_path.name},
                wait=True
            )
            print("done")
        except Exception as e:
            print(f"error: {e}")

    print("\nIndexing complete!")


def search(query: str, collection_name: str = "trondheim_docs", top_k: int = 3):
    """Search for relevant pages using visual embeddings."""
    client = get_client()

    print(f"Searching: {query}")
    results = client.search(
        query=query,
        collection_name=collection_name,
        top_k=top_k
    )

    return results


def generate_answer(query: str, results, model: str = "openai/gpt-4o-mini"):
    """Generate answer using retrieved page images."""
    # If we have images, use vision model
    content = [{"type": "text", "text": f"Based on these document pages, answer: {query}"}]

    for result in results.results:
        if result.img_base64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{result.img_base64}"}
            })
            content.append({
                "type": "text",
                "text": f"[{result.document_name}, Page {result.page_number}]"
            })

    response = requests.post(
        f"{OPENROUTER_BASE}/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 1000
        },
        timeout=120
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def main():
    parser = argparse.ArgumentParser(description="Visual RAG with ColiVara (ColPali API)")
    subparsers = parser.add_subparsers(dest="command")

    # Index command
    index_parser = subparsers.add_parser("index", help="Index PDFs")
    index_parser.add_argument("--pdf-dir", type=Path, default=Path("docs"))
    index_parser.add_argument("--collection", type=str, default="trondheim_docs")

    # Query command
    query_parser = subparsers.add_parser("query", help="Query documents")
    query_parser.add_argument("question", nargs="?")
    query_parser.add_argument("-k", "--top-k", type=int, default=3)
    query_parser.add_argument("--collection", type=str, default="trondheim_docs")
    query_parser.add_argument("--no-generate", action="store_true")
    query_parser.add_argument("--model", default="openai/gpt-4o-mini")
    query_parser.add_argument("-i", "--interactive", action="store_true")

    # List command
    list_parser = subparsers.add_parser("list", help="List collections")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "index":
        index_documents(args.pdf_dir, args.collection)

    elif args.command == "list":
        client = get_client()
        collections = client.list_collections()
        print("Collections:")
        for c in collections:
            print(f"  - {c.name}")

    elif args.command == "query":
        if not args.question and not args.interactive:
            query_parser.print_help()
            return

        def do_search(q: str):
            print(f"\n{'='*60}")
            print(f"Query: {q}")
            print(f"{'='*60}\n")

            results = search(q, args.collection, args.top_k)

            print("Retrieved pages:")
            for r in results.results:
                print(f"  [{r.normalized_score:.3f}] {r.document_name} - Page {r.page_number}")

            if not args.no_generate and results.results:
                print("\nGenerating answer...")
                answer = generate_answer(q, results, args.model)
                print(f"\n{answer}\n")

        if args.interactive:
            print("ColiVara Visual RAG (type 'quit' to exit)")
            print("-" * 40)
            while True:
                try:
                    q = input("\nQuery: ").strip()
                    if q.lower() in ('quit', 'exit', 'q'):
                        break
                    if q:
                        do_search(q)
                except (KeyboardInterrupt, EOFError):
                    print("\nBye!")
                    break
        else:
            do_search(args.question)


if __name__ == "__main__":
    main()
