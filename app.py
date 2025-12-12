import streamlit as st
import PyPDF2
from groq import Groq
import httpx  # <--- Opsional: Kalau error timeout masih bandel

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="HR AI Reviewer", page_icon="👔")

# --- JUDUL & SIDEBAR ---
st.title("📄 Chat dengan CV (HR AI)")
st.caption("Upload CV kamu, lalu ngobrol sama AI tentang isinya!")

with st.sidebar:
    st.header("Konfigurasi")
    # Input API Key biar aman (atau bisa hardcode kalau untuk sendiri)
    api_key = st.text_input("Masukkan Groq API Key:", type="password")
    
    # Upload File
    uploaded_file = st.file_uploader("Upload file CV (PDF)", type=("pdf"))

# --- FUNGSI BACA PDF ---
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# --- LOGIKA UTAMA ---
if api_key and uploaded_file:
    # 1. Inisialisasi Client Groq
    client = Groq(
        api_key=api_key,
        timeout=120.0,  # <--- tambahkan ini satuannya detik
        )

    # 2. Baca PDF & Simpan di Session State (Agar tidak baca ulang terus)
    if "pdf_text" not in st.session_state:
        st.session_state.pdf_text = extract_text_from_pdf(uploaded_file)
        st.success("✅ PDF berhasil dibaca! Silakan chat di bawah.")

    # 3. Inisialisasi History Chat
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": f"""
            Kamu adalah Asisten HRD yang santai.
            Tugasmu menjawab pertanyaan user berdasarkan isi dokumen CV ini:
            ---
            {st.session_state.pdf_text}
            ---
            Jawab dengan ringkas dan to-the-point.
            """}
        ]

    # 4. Tampilkan Chat History (Kecuali System Prompt)
    for message in st.session_state.messages:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # 5. Input Chat User
    if prompt := st.chat_input("Tanya sesuatu tentang CV ini..."):
        # Tampilkan pesan user
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Kirim ke AI
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=st.session_state.messages,
                temperature=0.7
            )
            ai_reply = response.choices[0].message.content
            
            # Tampilkan balasan AI
            st.chat_message("assistant").markdown(ai_reply)
            st.session_state.messages.append({"role": "assistant", "content": ai_reply})
            
        except Exception as e:
            #st.error(f"Error: {e}")
            # Ganti pesan error merah yang seram jadi lebih santai
            if "timed out" in str(e).lower():
                st.warning("⏳ AI-nya lagi mikir keras nih (Koneksi timeout). Coba tanya lagi ya!")
            else:
                st.error(f"Error: {e}")

elif not api_key:
    st.warning("👈 Masukkan API Key dulu di menu sebelah kiri!")
elif not uploaded_file:
    st.info("👈 Upload file PDF CV kamu dulu ya!")