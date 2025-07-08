# chatbot_rag_faiss_required_upload.py

import os
import streamlit as st
import fitz  # PyMuPDF
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv

# ----------------- 0. Load API Key từ .env -----------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY không tìm thấy trong .env. Vui lòng cấu hình trước.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# ----------------- 1. UI -----------------
st.set_page_config(page_title="Chatbot Chính sách Công ty", page_icon="🤖")
st.header("🤖 Chatbot Chính sách Công ty (Powered by Gemini)")
st.subheader("📄 Vui lòng tải lên file PDF để bắt đầu trò chuyện.")

uploaded_files = st.file_uploader("Tải lên các file PDF", type=["pdf"], accept_multiple_files=True)

# ----------------- 2. Trích xuất nội dung từ PDF -----------------
def extract_text_from_pdfs(files):
    documents = []
    for file in files:
        pdf_doc = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page in pdf_doc:
            text += page.get_text()
        documents.append(text)
    return documents

# ----------------- 3. Tạo Vector Store -----------------
@st.cache_resource
def get_vector_store_from_pdfs(files):
    texts = extract_text_from_pdfs(files)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.from_texts(texts, embeddings)

# Nếu không có file, dừng chương trình
if not uploaded_files:
    st.warning("⚠️ Vui lòng upload ít nhất một file PDF để sử dụng chatbot.")
    st.stop()

# Tạo vector store từ file đã upload
vector_store = get_vector_store_from_pdfs(uploaded_files)

# ----------------- 4. Khởi tạo mô hình Gemini & Chain -----------------
llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.2)

prompt_template = """
Bạn là một trợ lý AI thông minh và thân thiện. Hãy trả lời các câu hỏi dựa trên nội dung của tài liệu được cung cấp dưới đây. 

Nếu thông tin cần thiết không được nêu rõ trong tài liệu, bạn có thể dùng kiến thức chung hoặc suy luận logic từ dữ kiện đã có trong tài liệu để đưa ra câu trả lời hợp lý.

Hãy đảm bảo câu trả lời rõ ràng, mạch lạc, dễ hiểu và chính xác nhất có thể.

Ngữ cảnh:
{context}

Câu hỏi:
{question}

Trả lời:
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
qa_chain = load_qa_chain(llm=llm, chain_type="refine", prompt=prompt)

# ----------------- 5. Hàm trả lời câu hỏi -----------------
def get_gemini_response(question):
    docs = vector_store.similarity_search(question, k=4)
    response = qa_chain({"input_documents": docs, "question": question})
    return response["output_text"]

# ----------------- 6. Giao diện Chat -----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("💬 Bạn muốn hỏi gì?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🤔 Đang tìm câu trả lời..."):
            response = get_gemini_response(prompt)
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

# ----------------- 7. Thông tin file đã upload -----------------
if uploaded_files:
    st.markdown("---")
    st.success(f"📚 Đã upload {len(uploaded_files)} file:")
    for f in uploaded_files:
        st.markdown(f"- {f.name}")

st.markdown("---")
st.caption("💡 Chatbot này sử dụng Gemini và LangChain để trả lời dựa trên nội dung file PDF bạn đã cung cấp.")
