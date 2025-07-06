# chatbot_rag_faiss.py

import os
import streamlit as st
import fitz  # PyMuPDF để đọc file PDF
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

# ----------------- 1. UI: Cho phép người dùng upload PDF -----------------
st.set_page_config(page_title="Chatbot Chính sách Công ty", page_icon="🤖")
st.header("🤖 Chatbot Chính sách Công ty (Powered by Gemini)")
st.subheader("Tải file PDF về chính sách và hỏi chatbot về nội dung bên trong.")

uploaded_files = st.file_uploader("📄 Tải lên các file PDF", type=["pdf"], accept_multiple_files=True)

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

# ----------------- 3. Tạo Vector Store từ PDF -----------------
@st.cache_resource
def get_vector_store_from_pdfs(files):
    if not files:
        return None
    texts = extract_text_from_pdfs(files)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(texts, embeddings)
    return vector_store

# Nếu người dùng không upload, sử dụng dữ liệu mặc định (ví dụ mẫu)
default_docs = [
    "Chính sách nghỉ phép: Mỗi nhân viên được hưởng 15 ngày nghỉ phép có lương mỗi năm. Các ngày nghỉ phép không sử dụng sẽ được chuyển sang năm sau tối đa 5 ngày.",
    "Quy trình xin nghỉ phép: Nhân viên phải nộp đơn xin nghỉ phép qua hệ thống nội bộ ít nhất 3 ngày làm việc trước ngày nghỉ dự kiến. Phê duyệt cuối cùng thuộc về quản lý trực tiếp.",
    "Chính sách làm thêm giờ: Mọi công việc làm thêm giờ phải được quản lý trực tiếp phê duyệt trước. Tiền làm thêm giờ được tính theo hệ số 1.5 lần lương cơ bản cho các ngày trong tuần và 2.0 lần cho cuối tuần/ngày lễ.",
    "Quy tắc làm việc từ xa: Nhân viên có thể làm việc từ xa tối đa 2 ngày mỗi tuần, với sự đồng ý của quản lý. Cần đảm bảo kết nối internet ổn định và môi trường làm việc phù hợp.",
    "Chính sách bảo mật thông tin: Mọi thông tin khách hàng và dữ liệu nội bộ công ty là tài sản mật. Nghiêm cấm chia sẻ hoặc tiết lộ thông tin này cho bên thứ ba dưới mọi hình thức.",
    "Nghỉ không phép: sẽ bị trừ 1 ngày lương",
    "nghỉ quá 80% ngày công trong tháng sẽ không được tính lương"
]

@st.cache_resource
def get_default_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.from_texts(default_docs, embeddings)

# Chọn vector store tùy theo tình huống
vector_store = get_vector_store_from_pdfs(uploaded_files) if uploaded_files else get_default_vector_store()

# ----------------- 4. Khởi tạo mô hình Gemini & QA Chain -----------------
llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.2)

prompt_template = """
Bạn là một trợ lý thân thiện, chỉ trả lời các câu hỏi dựa trên thông tin được cung cấp trong ngữ cảnh sau.
Nếu thông tin không có trong ngữ cảnh, vui lòng nói rằng bạn không thể tìm thấy thông tin đó.
Bạn KHÔNG được trả lời bất kỳ câu hỏi nào nằm ngoài phạm vi của ngữ cảnh này.
Luôn cung cấp câu trả lời rõ ràng và ngắn gọn.

Ngữ cảnh:
{context}

Câu hỏi:
{question}

Trả lời:
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
qa_chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)

# ----------------- 5. Hàm trả lời câu hỏi -----------------
def get_gemini_response(question):
    if not vector_store:
        return "Không có dữ liệu để tìm kiếm. Vui lòng tải lên file PDF."
    docs = vector_store.similarity_search(question, k=2)
    response = qa_chain({"input_documents": docs, "question": question})
    return response["output_text"]

# ----------------- 6. Giao diện Chat -----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị lịch sử chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input người dùng
if prompt := st.chat_input("💬 Bạn muốn hỏi gì?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🤔 Đang tìm câu trả lời..."):
            response = get_gemini_response(prompt)
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

# Hiển thị file đã upload
if uploaded_files:
    st.markdown("---")
    st.success(f"📚 Đã upload {len(uploaded_files)} file:")
    for f in uploaded_files:
        st.markdown(f"- {f.name}")

st.markdown("---")
st.caption("💡 Chatbot này sử dụng Gemini và LangChain để trả lời dựa trên nội dung file PDF bạn đã cung cấp.")
