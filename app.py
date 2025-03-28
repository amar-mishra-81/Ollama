import streamlit as st
import ollama
from PIL import Image
import io
import base64
import PyPDF2

# Function to encode image to base64
def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

# Streamlit UI
st.set_page_config(page_title="Interactive AI Demo", page_icon="🤖", layout="wide")
st.title("Interactive AI Demo with Text, Images, and PDFs")
st.markdown("Explore AI capabilities by entering text, uploading images, or PDFs. Powered by Ollama (LLaVA).")

# Sidebar for instructions and settings
with st.sidebar:
    st.header("How to Use")
    st.write("""
    1. Type a question or prompt in the text box.
    2. Optionally upload an image or PDF.
    3. Click 'Get Response' or watch the live preview (if enabled).
    4. Enjoy the AI's response!
    """)
    live_preview = st.checkbox("Enable Live Preview", value=False)
    st.info("Note: Ensure 'ollama serve' is running locally.")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Text input
    text_input = st.text_area("Your Prompt", "Tell me about this content...", height=150)
    
    # File uploader
    uploaded_file = st.file_uploader("Upload an Image or PDF (optional)", type=["jpg", "png", "jpeg", "pdf"])

with col2:
    # Response display
    st.subheader("AI Response")
    response_container = st.empty()

# Function to get response from Ollama
def get_ollama_response(text, file=None):
    try:
        ollama.list()  # Check if Ollama is running
        if file:
            file_type = file.type
            if file_type == "application/pdf":
                pdf_text = extract_text_from_pdf(file)
                response = ollama.generate(
                    model="llava",
                    prompt=f"{text}\n\nPDF Content:\n{pdf_text[:2000]}"  # Limit PDF text to avoid overload
                )
                return response["response"], pdf_text[:500] + "..." if len(pdf_text) > 500 else pdf_text
            else:
                image = Image.open(file)
                img_base64 = image_to_base64(image)
                response = ollama.chat(
                    model="llava",
                    messages=[{"role": "user", "content": text, "images": [img_base64]}]
                )
                return response["message"]["content"], image
        else:
            response = ollama.generate(model="llava", prompt=text)
            return response["response"], None
    except Exception as e:
        return f"Error: {str(e)}. Is Ollama running on 127.0.0.1:11434?", None

# Button for manual submission
if st.button("Get Response") or (live_preview and text_input):
    with st.spinner("Processing..."):
        result, extra_content = get_ollama_response(text_input, uploaded_file)
        response_container.markdown(f"**Response:** {result}")
        
        # Display extra content (image or PDF text preview)
        if extra_content:
            if isinstance(extra_content, str):
                st.markdown("**PDF Preview:**")
                st.text(extra_content)
            elif isinstance(extra_content, Image.Image):
                st.image(extra_content, caption="Uploaded Image", use_column_width=True)

# Footer
st.markdown("---")
st.write("Built with ❤️ using Streamlit and Ollama | Demo by [Your Name]")