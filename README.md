# 📄 Resume Evaluation with Streamlit + NLP

This project is a Resume Evaluation web application built with **Streamlit**, utilizing **NLP**, **PDF parsing**, and **semantic similarity** techniques. It extracts structured data from PDF resumes and evaluates how well a candidate’s profile matches a specific job role (e.g., Web Developer, Data Scientist, etc.).

---

## 🚀 Features

- 📄 Read and clean raw text data from uploaded PDF resume files
- 🧠 Extract important details: Name, Email, Phone, Address, Skills, Experience, and Profile Summary
- 📊 Evaluate resumes using **SentenceTransformer** embeddings and cosine similarity
- 🔎 Recommend job roles based on detected skills (Web Dev, Data Science, Android, iOS, UI/UX)
- 🖥️ User-friendly Streamlit interface for uploading and viewing analysis results

---

## 🛠️ Tech Stack

- Python
- Streamlit
- spaCy
- Sentence Transformers (HuggingFace)
- PDFMiner.six
- wordninja
- Regex for text processing

---
## Screenshots

![resume ui](./images/resume.png)
![resume ui](./images/resume2.png)

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/resume-evaluation.git
cd resume-evaluation

# Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
# 📄 Resume Evaluation with Streamlit + NLP

This project is a Resume Evaluation web application built with **Streamlit**, utilizing **NLP**, **PDF parsing**, and **semantic similarity** techniques. It extracts structured data from PDF resumes and evaluates how well a candidate’s profile matches a specific job role (e.g., Web Developer, Data Scientist, etc.).

---

## 🚀 Features

- 📄 Read and clean raw text data from uploaded PDF resume files
- 🧠 Extract important details: Name, Email, Phone, Address, Skills, Experience, and Profile Summary
- 📊 Evaluate resumes using **SentenceTransformer** embeddings and cosine similarity
- 🔎 Recommend job roles based on detected skills (Web Dev, Data Science, Android, iOS, UI/UX)
- 🖥️ User-friendly Streamlit interface for uploading and viewing analysis results

---

## 🛠️ Tech Stack

- Python
- Streamlit
- spaCy
- Sentence Transformers (HuggingFace)
- PDFMiner.six
- wordninja
- Regex for text processing

---
## Screenshots
![Resume UI](./images/resume.png)
![Resume UI](./images/resume2.png)

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/resume-evaluation.git
cd resume-evaluation

# Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
