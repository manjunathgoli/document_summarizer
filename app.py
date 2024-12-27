import os
import streamlit as st #python lib for creating interactive web application for data science & machine learning
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
import torch #lib helps to do tensor operations & dynamic computations
import base64 #lib which helps to view pdfs

# Ensure the data directory exists
os.makedirs("data", exist_ok=True)

# Model and tokenizer setup with offloading folder
checkpoint = "LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(
    checkpoint, 
    device_map='auto', #selects cpu, gpu
    torch_dtype=torch.float32, #text is converted to digits and allocate 32bits
    offload_folder="./offload"
)

def file_preprocessing(file):
    """Preprocess the PDF file by splitting into smaller chunks."""
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(pages)
    return [text.page_content for text in texts]

def summarize_texts(input_texts, summarizer):
    """Iteratively summarize chunks and condense into a final summary."""
    intermediate_summaries = []
    for chunk in input_texts:
        try:
            result = summarizer(chunk, max_length=500, min_length=50, truncation=True)
            intermediate_summaries.append(result[0]['summary_text'])
        except Exception as e:
            st.error(f"Error summarizing chunk: {str(e)}")
    
    # Final summarization of all intermediate summaries
    final_summary = " ".join(intermediate_summaries)
    result = summarizer(final_summary, max_length=500, min_length=50, truncation=True)
    return result[0]['summary_text']

def llm_pipeline(filepath):
    """Summarize the content of a PDF file."""
    summarizer = pipeline(
        "summarization",
        model=base_model,
        tokenizer=tokenizer
    )
    input_texts = file_preprocessing(filepath)
    summary = summarize_texts(input_texts, summarizer)
    return summary

@st.cache_data
def displayPDF(file):
    """Display a PDF file within the Streamlit app."""
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

st.set_page_config(layout='wide', page_title="Summarization App")

def main():
    """Main function to run the Streamlit app."""
    st.title('Document Summarization App using Language Model')
    
    uploaded_file = st.file_uploader("Upload your PDF File", type=['pdf'])
    
    if uploaded_file is not None:
        if st.button("Summarize"):
            col1, col2 = st.columns(2)
            filepath = f"data/{uploaded_file.name}"
            
            # Save uploaded file
            with open(filepath, 'wb') as temp_file:
                temp_file.write(uploaded_file.read())
            
            # Display PDF and summary
            with col1:
                st.info("Uploaded PDF File")
                displayPDF(filepath)
                
            with col2:
                st.info("Summarization is below")
                summary = llm_pipeline(filepath)
                st.success(summary)

if __name__ == "__main__":
    main()
