import pdfplumber
import openai
from pinecone import Pinecone, ServerlessSpec
import os
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

# Directory to store extracted images
IMAGE_DIR = "extracted_images"
os.makedirs(IMAGE_DIR, exist_ok=True)


def extract_text(pdf_path):
    """Extract text and images from the PDF."""
    text_chunks = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            # Extract text
            text = page.extract_text()
            if text:
                text_chunks.append((i, text))  # Store with page number
        print("Text extracted successfully!")

    return text_chunks


def extract_images(pdf_path):
    """Extract images from the PDF and save them."""
    image_metadata = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            for img_idx, img in enumerate(page.images):
                try:
                    # Extract the raw image data
                    if "stream" in img:
                        img_data = img["stream"].get_rawdata()
                        image = Image.open(BytesIO(img_data))

                        # Ensure the image is in a valid format
                        if image.mode in ["1", "L", "P"]:  # Convert non-RGB images
                            image = image.convert("RGB")

                        img_filename = f"{IMAGE_DIR}/page_{i}_img_{img_idx}.png"
                        image.save(img_filename)

                        # Store image metadata
                        image_metadata.append({
                            "page": i,
                            "image_path": img_filename,
                            "bbox": img.get("bbox", None)  # Bounding box if available
                        })

                        print(f"Extracted image from page {i}, saved as {img_filename}")

                    else:
                        print(f"No valid image stream found on page {i}")

                except Exception as e:
                    print(f"Error processing image on page {i}: {e}")

    return image_metadata


def get_openai_embedding(text):
    """Generate embeddings using OpenAI's text-embedding-3-small"""
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=[text]  # OpenAI requires input as a list
    )
    return response.data[0].embedding


def store_data_in_pinecone(text_chunks, image_metadata):
    """Store extracted text and image metadata in Pinecone."""
    for page_num, chunk in text_chunks:
        embedding = get_openai_embedding(chunk)  # Get text embedding
        related_images = [img["image_path"] for img in image_metadata if img["page"] == page_num]

        metadata = {
            "text": chunk,
            "page": page_num,
            "images": related_images  # Store image file paths
        }

        index.upsert([(f"page-{page_num}", embedding, metadata)])
        print(f"Stored page {page_num} with text & images in Pinecone.")


# Run the pipeline
if __name__ == "__main__":
    pdf_path = "path_to_your_file.pdf"
    text_chunks = extract_text(pdf_path)
    image_metadata = extract_images(pdf_path)
    store_data_in_pinecone(text_chunks, image_metadata)

    print("PDF text and images have been processed and stored in Pinecone.")
