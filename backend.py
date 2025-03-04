import openai
from pinecone import Pinecone, ServerlessSpec
import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# Load API keys from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

# Initialize Flask app
app = Flask(__name__)


def find_relevant_content(query, top_k=3):
    """Finds relevant content and image references in Pinecone for a given user query."""
    # Generate embedding for the user query
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=[query]
    )
    query_embedding = response.data[0].embedding

    # Search Pinecone for relevant documents
    search_results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)

    # Extract relevant text chunks & images from metadata
    relevant_chunks = []
    relevant_images = []

    for match in search_results["matches"]:
        metadata = match["metadata"]
        if "text" in metadata:
            relevant_chunks.append(metadata["text"])
        if "images" in metadata:  # Ensure we capture image paths
            relevant_images.extend(metadata["images"])

    print("RELEVANT CHUNKS")
    print(relevant_chunks)
    print("RELEVANT IMAGES")
    print(relevant_images)

    return relevant_chunks, list(set(relevant_images))


def generate_response_with_gpt(user_query, relevant_text, images):
    """Generates a response using OpenAI GPT-4 with relevant text and images."""
    context = "\n".join(relevant_text)  # Combine all retrieved text chunks

    # Construct prompt for GPT-4
    prompt = f"""
    You are an AI assistant that helps users understand how to use an application.
    Answer the following user query based on the provided documentation.

    User Query: {user_query}

    Relevant Documentation:
    {context}

    If applicable, include references to the following images:
    {', '.join(images) if images else 'No relevant images found'}

    Provide a clear and concise answer preferably with bullet points or steps. 
    Do not include paths to the images if they exist.
    """

    # Get response from GPT-4
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


@app.route("/ask", methods=["POST"])
def handle_user_query():
    """API endpoint to handle user queries."""
    data = request.json
    user_query = data.get("query")

    if not user_query:
        return jsonify({"error": "Query is required"}), 400

    # Retrieve relevant content and images
    relevant_text, relevant_images = find_relevant_content(user_query)

    # Generate a response
    final_response = generate_response_with_gpt(user_query, relevant_text, relevant_images)

    return jsonify({"response": final_response, "images": relevant_images})


if __name__ == "__main__":
    app.run(debug=True)
