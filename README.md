# 📄 Document Summarization App 📝

Welcome to the **Document Summarization App**! This tool allows you to upload a PDF document and get a concise summary using advanced language models. Perfect for students, professionals, and anyone who needs quick insights from lengthy documents!

---

## 🚀 Features

- 📂 **Upload PDF Files**: Easily upload your document for summarization.
- ✂️ **Chunk Processing**: Breaks down long documents into manageable sections for effective summarization.
- 🧠 **Advanced Summarization**: Uses **LaMini-Flan-T5-248M** for high-quality summaries.
- 🌐 **Web Interface**: Built with Streamlit for a user-friendly experience.

---

## 🛠️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/<your-repo-name>/document-summarization-app.git
cd document-summarization-app
```
### 2. Create a Virtual Environment 🐍
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
### 3. Install Dependencies 📦
```bash
pip install -r requirements.txt
```
### 4. Clone the LaMini-Flan-T5-248M Model
```bash
git clone https://huggingface.co/MBZUAI/LaMini-Flan-T5-248M ./LaMini-Flan-T5-248M
```
### 🖥️ Run the Application
```bash
streamlit run app.py
```
### 📂 Project Structure
```bash
.
├── app.py                   # Main application file
├── data/                    # Folder for uploaded PDFs
├── requirements.txt         # Python dependencies
├── README.md                # This file!
├── LaMini-Flan-T5-248M/     # Cloned language model
```
### ✨ Example Usage
 1. Upload a PDF file from your local system.

 1. View the document preview on the left panel.
 3. Read the summarized text on the right panel.