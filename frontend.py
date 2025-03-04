import streamlit as st
import requests
from PIL import Image

# Backend API URL
BACKEND_URL = "http://127.0.0.1:5000/ask"  # Change this if backend is hosted remotely

st.set_page_config(page_title="User Guide AI", layout="wide")

# App Title
st.title("📘 User Guide AI - Ask Your Questions")

# User Query Input
user_query = st.text_input("Enter your question:", placeholder="How do I add a new user?")

# Button to Submit Query
if st.button("Ask"):
    if user_query.strip():
        with st.spinner("Searching the documentation..."):
            # Send user query to backend
            response = requests.post(BACKEND_URL, json={"query": user_query})

            if response.status_code == 200:
                data = response.json()
                answer = data["response"]
                images = data.get("images", [])

                # Display Response
                st.subheader("🔹 Response:")
                st.write(answer)

                # Display Relevant Images (if available)
                if images:
                    st.subheader("🖼️ Reference Images:")
                    cols = st.columns(len(images))  # Create columns for images
                    for i, img_path in enumerate(images):
                        try:
                            img = Image.open(img_path)
                            cols[i].image(img, caption=f"Reference Image {i + 1}", use_column_width=True)
                        except Exception as e:
                            st.error(f"Error loading image: {e}")
                else:
                    st.info("No relevant images found.")
            else:
                st.error("Failed to get response from backend. Please try again.")
    else:
        st.warning("Please enter a question before submitting.")
