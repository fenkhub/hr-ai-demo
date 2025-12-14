import streamlit as st
import os
import tempfile
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
# Ganti baris import chains yang lama dengan ini:
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Advanced HR RAG", page_icon="🧠")
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


st.title("🧠 HR AI: Vector Database Edition")
st.caption("Sekarang AI membaca CV menggunakan sistem RAG canggih (Embeddings + Vector DB).")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Konfigurasi")
    api_key = st.text_input("Masukkan Groq API Key:", type="password")
    
    st.divider()
    mode_hr = st.radio(
        "Style Wawancara:",
        ["😇 HR Ramah", "😈 HR Jaksel (Savage)"]
    )
    
    st.divider()
    uploaded_file = st.file_uploader("Upload CV (PDF)", type=("pdf"))

    if st.sidebar.button("Hapus Chat"):
        st.session_state.messages = [] # Kosongkan memori
        st.rerun() # Refresh halaman otomatis

# --- FUNGSI PROSES PDF KE VECTOR DB ---
def process_pdf_to_vector_db(uploaded_file):
    # 1. Simpan file sementara (LangChain butuh path file asli)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    # 2. Load PDF
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()

    # 3. Pecah Teks (Chunking)
    # Kita pecah jadi potongan 1000 karakter, dengan overlap 200 biar konteks nyambung
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(docs)

    # 4. Buat Embeddings (Ubah teks jadi angka)
    # Menggunakan model open-source gratis 'all-MiniLM-L6-v2'
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 5. Masukkan ke Vector Database (FAISS)
    vectors = FAISS.from_documents(final_documents, embeddings)
    
    # Hapus file sementara
    os.remove(tmp_path)
    return vectors

# --- LOGIKA UTAMA ---
if api_key and uploaded_file:
    
    # Inisialisasi Vector DB (Hanya sekali saat upload)
    if "vector_store" not in st.session_state:
        with st.spinner("Sedang memecah dokumen & membuat Index Vector... (Agak lama di awal)"):
            try:
                st.session_state.vector_store = process_pdf_to_vector_db(uploaded_file)
                st.success("✅ Dokumen berhasil di-index ke Vector DB!")
            except Exception as e:
                st.error(f"Gagal memproses PDF: {e}")

    # Siapkan Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Tampilkan Chat
    for message in st.session_state.messages:
        avatar = "👤" if message["role"] == "user" else ("😇" if "Ramah" in mode_hr else "😈")
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    # Input User
    if prompt := st.chat_input("Tanya sesuatu tentang CV ini..."):
        st.chat_message("user", avatar="👤").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # --- PROSES RAG LANGCHAIN ---
        try:
            # 1. Setup LLM (Groq)
            llm = ChatGroq(
                groq_api_key=api_key, 
                model_name="llama-3.1-8b-instant",
                temperature=0.7
            )

            # 2. Setup Prompt (Sesuai Persona)
            if "Ramah" in mode_hr:
                system_instruction = """
                Kamu adalah HRD Manager yang ramah dan suportif.
                Jawab pertanyaan berdasarkan konteks berikut:
                <context>
                {context}
                </context>
                Pertanyaan: {input}
                """
            else:
                system_instruction = """
                Kamu adalah HRD Jaksel yang galak, sinis, dan judgemental.
                Gunakan istilah gaul (literally, honestly, red flag).
                Jawab pertanyaan berdasarkan konteks berikut:
                <context>
                {context}
                </context>
                Pertanyaan: {input}
                """

            prompt_template = ChatPromptTemplate.from_template(system_instruction)

            # 3. Setup Chain (Rantai Proses)
            # Create Document Chain (Gabungin dokumen relevan ke prompt)
            document_chain = create_stuff_documents_chain(llm, prompt_template)
            
            # Create Retrieval Chain (Cari dokumen di Vector DB -> Kirim ke Document Chain)
            retriever = st.session_state.vector_store.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            # 4. Eksekusi
            with st.spinner("AI sedang mencari referensi di Vector DB..."):
                response = retrieval_chain.invoke({"input": prompt})
            
            ai_reply = response['answer']

            # Tampilkan Hasil
            icon_hr = "😇" if "Ramah" in mode_hr else "😈"
            st.chat_message("assistant", avatar=icon_hr).markdown(ai_reply)
            st.session_state.messages.append({"role": "assistant", "content": ai_reply})

        except Exception as e:
            st.error(f"Error: {e}")

elif not api_key:
    st.warning("👈 Masukkan API Key dulu bos!")
