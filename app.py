import streamlit as st
import PyPDF2
from groq import Groq

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="HR AI Reviewer", page_icon="🎭")

# --- BADGE PENGUNJUNG (VERSI BARU: Visitor Badge) ---
st.markdown(
    """
    <div style="text-align: center;">
        <h3>Pengunjung: <img src="https://visitor-badge.laobi.icu/badge?page_id=hr_ai_demo_streamlit_app" alt="visitors"></h3>
    </div>
    """,
    unsafe_allow_html=True
)

# ... lanjut ke judul & sidebar ...


# --- CSS BIAR TAMPILAN LEBIH BERSIH (Opsional) ---
st.markdown("""
<style>
    .stChatMessage {border-radius: 10px; padding: 10px;}
</style>
""", unsafe_allow_html=True)

# --- JUDUL & SIDEBAR ---
st.title("🎭 Chat dengan HR AI (Bipolar Mode)")
st.caption("Upload CV, lalu pilih: Mau HR yang memuji atau HR yang menjatuhkan mental?")

with st.sidebar:
    st.header("⚙️ Konfigurasi")
    
    # 1. Input API Key
    api_key = st.text_input("Masukkan Groq API Key:", type="password")
    
    # 2. Pilihan Mode HR (Switching Persona)
    st.divider()
    st.subheader("Pilih Kepribadian HR:")
    mode_hr = st.radio(
        "Style Wawancara:",
        ["😇 HR Ramah (Profesional)", "😈 HR Jaksel (Savage/Galak)"]
    )
    
    st.divider()
    
    # 3. Upload File
    uploaded_file = st.file_uploader("Upload file CV (PDF)", type=("pdf"))

# --- FUNGSI BACA PDF ---
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# --- DEFINISI SYSTEM PROMPT (PERSONA) ---
def get_system_prompt(mode, cv_text):
    if mode == "😇 HR Ramah (Profesional)":
        return f"""
        Kamu adalah HRD Manager yang sangat profesional, ramah, suportif, dan keibuan/kebapakan.
        Tugasmu: Review CV kandidat ini dan jawab pertanyaan user dengan sopan.
        Gaya Bicara: Formal, menggunakan "Anda", memberikan semangat, dan konstruktif.
        Selalu cari sisi positif dari kandidat, meskipun skill-nya kurang.
        
        DATA CV KANDIDAT:
        {cv_text}
        """
    else: # Mode Galak
        return f"""
        Kamu adalah HRD Manager "Anak Jaksel" yang sangat galak, sinis, judgemental, dan elitist.
        Tugasmu: Roasting (ejek) CV kandidat ini. Kamu benci kandidat yang skill-nya standar.
        Gaya Bicara:
        1. Campur Inggris-Indonesia (Jaksel)
        2. Campur Bahasa Indonesia dan Bahasa Inggris (Indonesian-English code-mixing) di setiap kalimat.
        3. Gunakan istilah gaul seperti: "literally", "honestly", "prefer", "vibe", "red flag", "big no", "lowkey", "makes sense", "culture fit", "cringe".
        3. Kamu menolak pelamar ini karena skill-nya terlalu dasar (cuma belajar Python seminggu di YouTube).
        4. Nadanya tidak marah, tapi meremehkan dengan gaya santai (passive-aggressive).
        5. Jangan sopan! Bersikaplah meremehkan. Kalau skill-nya jelek, bilang jelek langsung.       
        
        DATA CV KANDIDAT:
        {cv_text}
        """
# Gaya Bicara: Campur Inggris-Indonesia (Jaksel), pakai kata: 'literally', 'red flag', 'honestly', 'big no', 'cringe'.
# Jangan sopan! Bersikaplah meremehkan. Kalau skill-nya jelek, bilang jelek langsung.

# --- LOGIKA UTAMA ---
if api_key and uploaded_file:
    # 1. Inisialisasi Client (dengan Timeout lebih lama)
    client = Groq(api_key=api_key, timeout=120.0)

    # 2. Baca PDF Sekali Saja
    if "pdf_text" not in st.session_state:
        with st.spinner("Sedang membaca PDF..."):
            st.session_state.pdf_text = extract_text_from_pdf(uploaded_file)
        st.success("✅ PDF terbaca! Siap di-roasting atau dipuji.")

    # 3. Update System Prompt Real-time
    # Setiap kali user ganti radio button, prompt di memori (messages[0]) langsung diupdate
    current_system_prompt = get_system_prompt(mode_hr, st.session_state.pdf_text)
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "system", "content": current_system_prompt}]
    else:
        # Update prompt sistem di index ke-0 agar sesuai mode yg dipilih sekarang
        st.session_state.messages[0]["content"] = current_system_prompt

    # 4. Tampilkan Chat History
    for message in st.session_state.messages:
        if message["role"] != "system":
            # Ganti ikon avatar sesuai role
            avatar = "👤" if message["role"] == "user" else ("😇" if "Ramah" in mode_hr else "😈")
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

    # 5. Input Chat User
    if prompt := st.chat_input("Tanya pendapat HR tentang CV ini..."):
        # Tampilkan pesan user
        st.chat_message("user", avatar="👤").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Kirim ke AI
        try:
            # Pilihan Model: Bisa ganti ke 'llama-3.3-70b-versatile' kalau 8b kurang galak
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant", 
                messages=st.session_state.messages,
                temperature=0.8 # Agak tinggi biar kreatif galaknya
            )
            ai_reply = response.choices[0].message.content
            
            # Tampilkan balasan AI
            icon_hr = "😇" if "Ramah" in mode_hr else "😈"
            st.chat_message("assistant", avatar=icon_hr).markdown(ai_reply)
            st.session_state.messages.append({"role": "assistant", "content": ai_reply})
            
        except Exception as e:
            st.error(f"Error: {e}")

elif not api_key:
    st.warning("👈 Masukkan API Key dulu di menu sebelah kiri!")
elif not uploaded_file:
    st.info("👈 Upload file PDF CV kamu dulu ya!")
