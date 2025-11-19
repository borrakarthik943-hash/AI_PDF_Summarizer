# AI_PDF_Summarizer

# 🎙️ EchoVerse — AI PDF Summarizer & Audiobook Generator

**Powered by IBM Granite + IBM Watson TTS + HuggingFace + Streamlit**

EchoVerse is an advanced AI application that converts PDF documents into concise, high-quality summaries and transforms them into audiobooks.
It uses **IBM Granite 4.0** for summarization and supports **IBM Watson Text-to-Speech** (with gTTS fallback), packaged inside an elegant, interactive Streamlit web interface.

---

## 🚀 Features

### ✅ **AI PDF Summarization**

* Upload any PDF (research papers, books, articles, reports)
* Extracts text automatically using `pdfplumber`
* Splits long PDFs into smart chunks
* Summarizes using **IBM Granite 4.0** models via Hugging Face
* Supports multiple summary styles:

  * **Concise**
  * **Bullet Points**
  * **Detailed**

### 🎧 **AI Audiobook Generation**

* Converts the summary into natural-sounding speech
* Supports two TTS engines:

  * **IBM Watson Text-to-Speech (recommended)**
  * **gTTS (free fallback)**

### 🎨 **Streamlit Web Interface**

* Clean, modern UI
* Side-by-side preview of extracted text & summary
* Audio playback + MP3 download
* Real-time progress indicators
* Metrics (word count, summary length, chunk count)

---

## 🧠 Technologies Used

| Component               | Technology                                     |
| ----------------------- | ---------------------------------------------- |
| **Summarization Model** | IBM Granite 4.0 (Hugging Face Transformers)    |
| **PDF Text Extraction** | pdfplumber                                     |
| **TTS (Primary)**       | IBM Watson Text-to-Speech                      |
| **TTS (Fallback)**      | Google gTTS                                    |
| **Frontend**            | Streamlit                                      |
| **Runtime**             | Python 3.10+                                   |
| **Deployment-ready**    | Works in Google Colab, local machine, or cloud |

---

## 📦 Installation

### 1️⃣ Install required dependencies

```bash
pip install streamlit torch transformers pdfplumber gTTS ibm-watson
```

### 2️⃣ (Optional) Install ngrok for Colab access

```bash
pip install pyngrok
```

---

## 🔑 IBM Watson Credentials (Optional but Recommended)

Set environment variables:

```bash
export IBM_WATSON_APIKEY="your_api_key"
export IBM_WATSON_URL="your_service_url"
```

Or enter them in the Streamlit sidebar during runtime.

---

## ▶️ Running the App

### **Local System**

```bash
streamlit run echoverse_pdf_summarizer.py
```

### **Google Colab**

Use:

```bash
!streamlit run echoverse_pdf_summarizer.py --server.enableCORS false --server.enableXsrfProtection false
```

Then expose the app with `ngrok`.

---

## 📁 Project Structure

```
EchoVerse/
│
├── echoverse_pdf_summarizer.py     # Main Streamlit application
├── README.md                        # Project documentation
└── requirements.txt                 # Dependencies
```

---

## 🖥️ How It Works (Pipeline)

1. **Upload PDF**
2. **Extract Text** using `pdfplumber`
3. **Chunk Long Content** automatically
4. **Summarize Each Chunk** using IBM Granite
5. **Merge Chunk Summaries** into a final summary
6. **Generate Audiobook** using IBM TTS or gTTS
7. **Preview + Download MP3**

---

## 🎯 Use Cases

* Turn research papers into audio summaries
* Convert academic PDFs to accessible audio
* Generate audiobook versions of reports
* Quickly digest long documents
* Build assistive tools for visually impaired users

---

## 🤝 Contributing

PRs are welcome!
Feel free to open issues, request features, or add new summarization models/voices.

---

## 📄 License

MIT License © 2025 EchoVerse AI

---

## ⭐ Acknowledgements

* **IBM Granite Models** (Hugging Face)
* **IBM Watson Text-to-Speech**
* **Google Text-to-Speech**
* **Streamlit Team**
* **Open-source community**

---

If you like this project, consider starring it ⭐ on GitHub!
