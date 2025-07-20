import os
import json
import streamlit as st
from dotenv import load_dotenv
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
import google.generativeai as genai

# ----------------- 0. Load ENV -----------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SHEET_ID = os.getenv("GOOGLE_SHEET_ID")
CREDENTIALS_JSON = os.getenv("GOOGLE_CREDENTIALS_JSON")

if not GOOGLE_API_KEY or not SHEET_ID or not CREDENTIALS_JSON:
    st.error("❌ Thiếu GOOGLE_API_KEY, GOOGLE_SHEET_ID hoặc GOOGLE_CREDENTIALS_JSON trong .env")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# ----------------- 1. Đọc Google Sheet -----------------
def extract_text_from_google_sheet():
    try:
        creds_dict = json.loads(CREDENTIALS_JSON)
    except Exception as e:
        st.error(f"❌ Lỗi khi parse GOOGLE_CREDENTIALS_JSON: {e}")
        st.stop()

    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)
    sheet = client.open_by_key(SHEET_ID).sheet1

    rows = sheet.get_all_values()
    if len(rows) <= 1:
        return ["Sheet không có dữ liệu đủ."]
    
    texts = [f"Hỏi: {r[0]}\nĐáp: {r[1]}" for r in rows[1:] if len(r) >= 2]
    return texts

# ----------------- 2. Vector Store -----------------
@st.cache_resource
def get_vector_store_from_sheet():
    texts = extract_text_from_google_sheet()
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.from_texts(texts, embeddings)

# ----------------- 3. Nút làm mới dữ liệu -----------------
if st.button("🔄 Làm mới dữ liệu từ Google Sheet"):
    st.cache_resource.clear()
    st.success("✅ Dữ liệu đã được làm mới.")
    st.rerun()

# ----------------- 4. LLM & Prompt -----------------
llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.2)

prompt_template = """
Bạn là một trợ lý AI thân thiện và hiểu rõ chính sách công ty. Hãy trả lời rõ ràng, dễ hiểu dựa trên dữ liệu sau:

{context}

Câu hỏi:
{question}

Trả lời:
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
qa_chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)

# ----------------- 5. Tạo Vector Store -----------------
vector_store = get_vector_store_from_sheet()

# ----------------- 6. Giao diện Chat -----------------
st.set_page_config(page_title="Chatbot Chính sách Công ty", page_icon="🤖")
st.header("🤖 Chatbot Chính sách Công ty (Google Sheets + Gemini)")
st.caption("💡 Dữ liệu được nạp từ Google Sheet chứa thông tin hỏi đáp chính sách.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("💬 Nhập câu hỏi của bạn..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🤔 Đang tìm câu trả lời..."):
            docs = vector_store.similarity_search(prompt, k=4)
            response = qa_chain({"input_documents": docs, "question": prompt})
            st.markdown(response["output_text"])
            st.session_state.messages.append({"role": "assistant", "content": response["output_text"]})

st.markdown("---")
st.caption("📚 Chatbot sử dụng LangChain + Gemini và dữ liệu từ Google Sheets.")
