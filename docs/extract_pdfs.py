from pathlib import Path
from docling.document_converter import DocumentConverter

docs_dir = Path(__file__).parent
txt_dir = docs_dir / "txt"
txt_dir.mkdir(exist_ok=True)

converter = DocumentConverter()

pdf_files = list(docs_dir.glob("*.pdf"))
print(f"Found {len(pdf_files)} PDF files")

for pdf_path in pdf_files:
    print(f"Processing: {pdf_path.name}")
    try:
        result = converter.convert(str(pdf_path))
        text = result.document.export_to_markdown()

        output_path = txt_dir / f"{pdf_path.stem}.txt"
        output_path.write_text(text)
        print(f"  -> Saved to {output_path.name}")
    except Exception as e:
        print(f"  -> Error: {e}")

print("Done!")
