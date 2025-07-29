import os
import json
import streamlit as st
import gspread
from dotenv import load_dotenv
from oauth2client.service_account import ServiceAccountCredentials
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
import google.generativeai as genai

# ----------------- 0. Load ENV -----------------
load_dotenv()
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
SHEET_ID = st.secrets["GOOGLE_SHEET_ID"]
GOOGLE_CRED = st.secrets["GOOGLE_CRED"]

if not GOOGLE_API_KEY or not SHEET_ID or not GOOGLE_CRED:
    st.error("❌ Thiếu GOOGLE_API_KEY, GOOGLE_SHEET_ID hoặc GOOGLE_CRED trong file .env")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# ----------------- 1. Đọc Google Sheet -----------------
def extract_text_from_google_sheet():
    try:
        creds_dict = json.loads(GOOGLE_CRED)
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        sheet = client.open_by_key(SHEET_ID).sheet1
    except Exception as e:
        st.error(f"❌ Lỗi khi kết nối Google Sheet: {e}")
        st.stop()

    rows = sheet.get_all_values()
    if len(rows) <= 1:
        return ["Sheet không có dữ liệu đủ."]

    texts = []
    for r in rows[1:]:
        if len(r) >= 1:
            main_text = r[0].strip()  # Cột A: nội dung chính
            # Duyệt qua các cột phụ từ cột B trở đi
            supplemental_parts = [cell.strip() for cell in r[1:] if cell.strip()]
            supplemental = "\n".join([f"- {text}" for text in supplemental_parts])
            
            if supplemental:
                combined = f"Nội dung chính: {main_text}\nThông tin bổ sung:\n{supplemental}"
            else:
                combined = f"Nội dung chính: {main_text}"
            texts.append(combined)

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
