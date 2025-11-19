# echoverse_pdf_summarizer.py
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pdfplumber
import tempfile
import os
import time
import math
import warnings

# Optional TTS libraries
from gtts import gTTS
try:
    from ibm_watson import TextToSpeechV1
    from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
    HAS_IBM_WATSON = True
except Exception:
    HAS_IBM_WATSON = False

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="📚 EchoVerse PDF Summarizer → Audiobook",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

device = "cuda" if torch.cuda.is_available() else "cpu"
device_label = "GPU 🚀" if device == "cuda" else "CPU 🖥️"

# ========== CSS ==========
st.markdown(
    """
    <style>
    .title {font-size:32px; font-weight:700; color:#ffffff; text-align:center;}
    .sub {color:#d0d7ff; text-align:center;}
    .box {background:white; padding:16px; border-radius:8px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="title">🎙 EchoVerse — PDF Summarizer → Audiobook</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub">Summarize PDF using IBM Granite 4.0 (Hugging Face) + TTS — Device: {device_label}</div>', unsafe_allow_html=True)
st.markdown("---")

# ========== MODEL LOADING ==========
@st.cache_resource(show_spinner=False)
def load_granite_model(model_id="ibm-granite/granite-4.0-h-350m"):
    """Load Granite tokenizer + model (Hugging Face)."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32 if device == "cpu" else torch.float16,
            device_map={"": device} if device == "cuda" else None,
            trust_remote_code=True
        )
        model.eval()
        return model, tokenizer
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None, None

# ========== UTILITIES ==========
def extract_text_from_pdf(uploaded_file):
    """Extract text from uploaded PDF using pdfplumber."""
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            pages_text = [p.extract_text() or "" for p in pdf.pages]
        full_text = "\n\n".join(pages_text).strip()
        return full_text
    except Exception as e:
        st.error(f"PDF extraction error: {e}")
        return ""

def chunk_text(text, max_chars=2500):
    """Naive chunking by characters to fit model context. Returns list of chunks."""
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + max_chars, text_len)
        # try to avoid breaking mid-sentence
        if end < text_len:
            last_period = text.rfind(".", start, end)
            if last_period != -1 and last_period - start > 200:
                end = last_period + 1
        chunks.append(text[start:end].strip())
        start = end
    return chunks

def summarize_with_granite(text, model, tokenizer, summary_style="concise", max_new_tokens=200):
    """
    Summarize a single text chunk with a prompt. Returns generated summary string.
    This method uses a simple prompt; for production refine prompts and safety checks.
    """
    if model is None or tokenizer is None:
        return "[model unavailable] " + text[:200]

    prompt_map = {
        "concise": "Summarize the following text concisely, focusing on main points:\n\n",
        "bullet": "Summarize the following text as concise bullet points:\n\n",
        "detailed": "Provide a detailed summary of the following text:\n\n"
    }
    prompt_prefix = prompt_map.get(summary_style, prompt_map["concise"])
    prompt = prompt_prefix + text.strip() + "\n\nSummary:"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.2,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id
        )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    # strip prompt from decoded text if present
    summary = decoded.replace(prompt, "").strip()
    return summary

def assemble_summaries(chunks, model, tokenizer, style):
    """Summarize each chunk and join them into a final summary."""
    summaries = []
    for i, c in enumerate(chunks):
        st.experimental_log("info", f"Summarizing chunk {i+1}/{len(chunks)} (len={len(c)})")
        s = summarize_with_granite(c, model, tokenizer, summary_style=style, max_new_tokens=180)
        summaries.append(s)
    # If multiple chunk summaries, produce an overall consolidation prompt
    if len(summaries) > 1:
        combined = "\n\n".join(summaries)
        consolidation_prompt = "Consolidate the following chunk summaries into a single coherent summary:\n\n" + combined + "\n\nFinal summary:"
        inputs = tokenizer(consolidation_prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=250, do_sample=False, temperature=0.2)
        consolidated = tokenizer.decode(out[0], skip_special_tokens=True).replace(consolidation_prompt, "").strip()
        return consolidated
    else:
        return summaries[0] if summaries else ""

def tts_with_gtts(text, filename):
    t = gTTS(text=text, lang="en", slow=False)
    t.save(filename)
    return filename

def tts_with_ibm_watson(text, filename, api_key, url, voice="en-US_AllisonV3Voice"):
    """
    Convert text to speech using IBM Watson TTS.
    Requires: ibm_watson package and valid IAM API key + url.
    Returns path to saved audio file, or None on failure.
    """
    try:
        authenticator = IAMAuthenticator(api_key)
        tts_service = TextToSpeechV1(authenticator=authenticator)
        tts_service.set_service_url(url)

        # Watson returns WAV or other formats; we request audio/mp3
        response = tts_service.synthesize(text, voice=voice, accept='audio/mp3').get_result()
        with open(filename, 'wb') as f:
            f.write(response.content)
        return filename
    except Exception as e:
        st.error(f"IBM Watson TTS error: {e}")
        return None

# ========== SIDEBAR ==========
with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown("---")
    model_id = st.text_input("Hugging Face model id", value="ibm-granite/granite-4.0-h-350m")
    summary_style = st.selectbox("Summary style", ["concise", "bullet", "detailed"])
    max_chars_per_chunk = st.number_input("Max chars per chunk (approx)", value=2500, min_value=1000, max_value=8000, step=500)
    tts_choice = st.selectbox("TTS Provider", ["IBM Watson (requires credentials)", "gTTS (fallback/free)"])
    if tts_choice.startswith("IBM"):
        st.markdown("Provide IBM Watson TTS credentials as env vars or below.")
        ibm_api_key = st.text_input("IBM TTS API Key (or set IBM_WATSON_APIKEY env var)", type="password")
        ibm_url = st.text_input("IBM TTS URL (or set IBM_WATSON_URL env var)", value=os.environ.get("IBM_WATSON_URL", ""))
        ibm_voice = st.text_input("IBM TTS voice (optional)", value="en-US_AllisonV3Voice")
    else:
        ibm_api_key = None
        ibm_url = None
        ibm_voice = None
    st.markdown("---")
    st.write(f"Device: **{device_label}**")
    st.caption("Model load will be cached. If updating model id, reload the app.")

# ========== UPLOAD PDF ==========
st.markdown("### 1) Upload PDF")
uploaded_file = st.file_uploader("Upload a PDF file to summarize", type=["pdf"])

if uploaded_file:
    with st.spinner("Extracting PDF text..."):
        raw_text = extract_text_from_pdf(uploaded_file)
    if not raw_text:
        st.error("No text could be extracted from this PDF. Try a different file.")
    else:
        st.success("Text extracted from PDF")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("#### PDF Preview (first 6K chars)")
            st.text_area("", value=raw_text[:6000], height=220, disabled=True)
            st.markdown("---")
            approx_words = len(raw_text.split())
            st.metric("Approx words in document", approx_words)

        with col2:
            st.markdown("#### Summarization & Audiobook")
            model, tokenizer = load_granite_model(model_id=model_id)
            if model is None or tokenizer is None:
                st.error("Model failed to load. Check model id or internet connection.")
            else:
                # Ready to summarize
                if st.button("🔎 Summarize & Generate Audiobook"):
                    # chunk
                    chunks = chunk_text(raw_text, max_chars=max_chars_per_chunk)
                    progress = st.progress(0)
                    status = st.empty()
                    try:
                        status.text("Summarizing chunks...")
                        combined_summary = assemble_summaries(chunks, model, tokenizer, summary_style)
                        progress.progress(60)
                        status.text("Preparing audio...")
                        # Shorten summary if extremely long (TTS limitations)
                        if len(combined_summary) > 15000:
                            combined_summary = combined_summary[:15000] + " ... (truncated for TTS)"

                        # choose tts provider
                        tmp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                        audio_path = tmp_audio.name
                        tmp_audio.close()

                        if tts_choice.startswith("IBM") and (ibm_api_key or os.environ.get("IBM_WATSON_APIKEY")) and (ibm_url or os.environ.get("IBM_WATSON_URL")) and HAS_IBM_WATSON:
                            api_key = ibm_api_key or os.environ.get("IBM_WATSON_APIKEY")
                            url = ibm_url or os.environ.get("IBM_WATSON_URL")
                            status.text("Converting to speech with IBM Watson TTS...")
                            audio_file = tts_with_ibm_watson(combined_summary, audio_path, api_key, url, voice=ibm_voice or "en-US_AllisonV3Voice")
                            if audio_file:
                                st.success("Audio generated with IBM Watson TTS")
                            else:
                                st.warning("IBM TTS failed — falling back to gTTS")
                                audio_file = tts_with_gtts(combined_summary, audio_path)
                        else:
                            status.text("Converting to speech with gTTS...")
                            audio_file = tts_with_gtts(combined_summary, audio_path)
                            st.success("Audio generated with gTTS")

                        progress.progress(100)
                        status.empty()

                        # Display results
                        st.markdown("### 📝 Summary")
                        st.write(combined_summary)

                        st.markdown("---")
                        st.markdown("### 🎧 Listen / Download")
                        if audio_file and os.path.exists(audio_file):
                            with open(audio_file, "rb") as f:
                                st.audio(f.read(), format="audio/mp3")
                            with open(audio_file, "rb") as f:
                                st.download_button("📥 Download MP3", f.read(), file_name="echoverse_summary.mp3", mime="audio/mp3")
                        else:
                            st.error("Audio file not available.")

                        # Stats
                        st.markdown("---")
                        c1, c2, c3 = st.columns(3)
                        c1.metric("PDF words", len(raw_text.split()))
                        c2.metric("Summary words", len(combined_summary.split()))
                        c3.metric("Chunks", len(chunks))

                    except Exception as e:
                        st.error(f"Processing error: {e}")
                        progress.empty()
