#!/usr/bin/env python3
"""
DeepSeek-OCR: Convert documents (PDF/PNG/JPG) to Markdown

Usage:
    uv run deepseek_ocr.py <input> [output.md]
    uvx --from . deepseek_ocr <input> [output.md]

Example:
    uv run deepseek_ocr.py paper.pdf
    uvx --from . deepseek_ocr document.png output.md
"""

import sys
import os
from pathlib import Path
import tempfile


def pdf_to_images(pdf_path):
    """Convert PDF pages to images"""
    import fitz  # PyMuPDF
    from PIL import Image
    import io

    pdf_document = fitz.open(pdf_path)
    images = []

    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        zoom = 2.0
        matrix = fitz.Matrix(zoom, zoom)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)

        img_data = pixmap.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        images.append(img)

    pdf_document.close()
    return images


def ocr_image(model, tokenizer, image_path, temp_dir):
    """Perform OCR on a single image"""
    prompt = "<image>\n<|grounding|>Convert the document to markdown. "
    output_path = temp_dir

    res = model.infer(
        tokenizer,
        prompt=prompt,
        image_file=str(image_path),
        output_path=output_path,
        base_size=1024,
        image_size=640,
        crop_mode=True,
        save_results=True,
        test_compress=True
    )

    # Read the result
    result_file = Path(output_path) / "result.mmd"
    if result_file.exists():
        with open(result_file, 'r', encoding='utf-8') as f:
            return f.read()
    return ""


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run deepseek_ocr.py <input> [output.md]")
        print("\nSupports: PDF, PNG, JPG")
        print("\nExample:")
        print("  uv run deepseek_ocr.py paper.pdf")
        print("  uvx --from . deepseek_ocr document.png output.md")
        sys.exit(1)

    pdf_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    else:
        output_path = Path(pdf_path).stem + ".md"

    if not Path(pdf_path).exists():
        print(f"Error: {pdf_path} not found")
        sys.exit(1)

    print(f"Converting {pdf_path} to Markdown...")
    print(f"Output: {output_path}")

    # Load model
    print("\nLoading DeepSeek-OCR model...")
    from transformers import AutoModel, AutoTokenizer
    import torch

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    model_name = 'deepseek-ai/DeepSeek-OCR'

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        attn_implementation='eager',
        trust_remote_code=True,
        use_safetensors=True
    )
    model = model.eval().cuda().to(torch.bfloat16)

    print("Model loaded successfully!")

    # Convert PDF to images
    print(f"\nConverting PDF to images...")
    images = pdf_to_images(pdf_path)
    print(f"Found {len(images)} pages")

    # Create output directory for images
    output_base = Path(output_path).stem
    output_dir = Path(output_path).parent / f"{output_base}_files"
    output_dir.mkdir(exist_ok=True)

    # Process each page
    markdown_output = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for i, img in enumerate(images):
            print(f"\nProcessing page {i+1}/{len(images)}...")

            # Save image temporarily
            img_path = Path(temp_dir) / f"page_{i+1}.png"
            img.save(img_path)

            # OCR the image
            ocr_temp_dir = Path(temp_dir) / f"output_{i+1}"
            ocr_temp_dir.mkdir(exist_ok=True)

            page_markdown = ocr_image(model, tokenizer, img_path, str(ocr_temp_dir))

            if page_markdown:
                markdown_output.append(f"<!-- Page {i+1} -->")
                markdown_output.append(page_markdown)
                markdown_output.append("")  # Empty line between pages

                # Copy images from temp directory to output directory
                images_src = Path(ocr_temp_dir) / "images"
                if images_src.exists():
                    images_dst = output_dir / f"page_{i+1}_images"
                    images_dst.mkdir(exist_ok=True, parents=True)

                    import shutil
                    for img_file in images_src.glob("*"):
                        shutil.copy2(img_file, images_dst / img_file.name)

    # Save output
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(markdown_output))

    print(f"\n✓ Conversion complete!")
    print(f"Output saved to: {output_path}")
    print(f"Images saved to: {output_dir}")


if __name__ == "__main__":
    main()
