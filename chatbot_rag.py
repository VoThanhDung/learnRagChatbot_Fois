# chatbot_rag.py

import os
import streamlit as st
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv

# Load environment variables (your API key)
load_dotenv()

# Get the Google API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found in environment variables. Please set it in a .env file.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)
print("Available models:")
for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        print(f"- {m.name} (supports generateContent)")
    else:
        print(f"- {m.name}")
# ----------------- 1. Chuẩn bị dữ liệu (từ documents.py hoặc nội dung trực tiếp) -----------------
company_policy_docs = [
    "Chính sách nghỉ phép: Mỗi nhân viên được hưởng 15 ngày nghỉ phép có lương mỗi năm. Các ngày nghỉ phép không sử dụng sẽ được chuyển sang năm sau tối đa 5 ngày.",
    "Quy trình xin nghỉ phép: Nhân viên phải nộp đơn xin nghỉ phép qua hệ thống nội bộ ít nhất 3 ngày làm việc trước ngày nghỉ dự kiến. Phê duyệt cuối cùng thuộc về quản lý trực tiếp.",
    "Chính sách làm thêm giờ: Mọi công việc làm thêm giờ phải được quản lý trực tiếp phê duyệt trước. Tiền làm thêm giờ được tính theo hệ số 1.5 lần lương cơ bản cho các ngày trong tuần và 2.0 lần cho cuối tuần/ngày lễ.",
    "Quy tắc làm việc từ xa: Nhân viên có thể làm việc từ xa tối đa 2 ngày mỗi tuần, với sự đồng ý của quản lý. Cần đảm bảo kết nối internet ổn định và môi trường làm việc phù hợp.",
    "Chính sách bảo mật thông tin: Mọi thông tin khách hàng và dữ liệu nội bộ công ty là tài sản mật. Nghiêm cấm chia sẻ hoặc tiết lộ thông tin này cho bên thứ ba dưới mọi hình thức."
]

# ----------------- 2. Tạo Embeddings và Vector Store -----------------
# Sử dụng GoogleGenerativeAIEmbeddings để tạo embeddings
@st.cache_resource
def get_vector_store():
    # Khởi tạo mô hình nhúng của Google
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Tạo vector store từ các tài liệu và embeddings
    # persist_directory: thư mục để lưu trữ dữ liệu của ChromaDB
    vector_store = Chroma.from_texts(company_policy_docs, embeddings, persist_directory="./chroma_db")
    
    # Lưu trữ vector store để không phải tạo lại mỗi lần chạy
    vector_store.persist()
    return vector_store

# Lấy hoặc tạo vector store
vector_store = get_vector_store()

# ----------------- 3. Thiết lập Gemini API và LangChain Chain -----------------
# Khởi tạo mô hình Gemini
llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.2) # temperature=0.2 để câu trả lời ít sáng tạo hơn

# Tạo prompt template để hướng dẫn Gemini
# {context} sẽ là nơi chứa các đoạn tài liệu liên quan được tìm thấy
# {question} là câu hỏi của người dùng
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

# Tạo LangChain PromptTemplate
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Tải chuỗi QA (Question Answering)
# chain_type="stuff": đưa tất cả các tài liệu tìm được vào một prompt duy nhất
# prompt=prompt: sử dụng prompt template đã định nghĩa
# llm=llm: sử dụng mô hình Gemini đã khởi tạo
qa_chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)

# ----------------- 4. Xây dựng hàm truy vấn RAG -----------------
def get_gemini_response(question):
    # Tìm kiếm các tài liệu liên quan nhất trong vector store
    # k=2: tìm 2 đoạn tài liệu gần nhất
    docs = vector_store.similarity_search(question, k=2) 
    
    # Chạy chuỗi QA với câu hỏi và các tài liệu liên quan
    response = qa_chain({"input_documents": docs, "question": question})
    return response["output_text"]

# ----------------- 5. Tạo giao diện Chatbot với Streamlit -----------------
st.set_page_config(page_title="Chatbot Chính sách Công ty", page_icon="🤖")

st.header("🤖 Chatbot Chính sách Công ty (Powered by Gemini)")
st.subheader("Hỏi tôi về chính sách nghỉ phép, làm thêm giờ, làm việc từ xa, và bảo mật thông tin.")

# Khởi tạo lịch sử chat nếu chưa có
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị lịch sử chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Xử lý input của người dùng
if prompt := st.chat_input("Bạn có câu hỏi gì về chính sách?"):
    # Thêm tin nhắn người dùng vào lịch sử
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Lấy phản hồi từ Gemini
    with st.chat_message("assistant"):
        with st.spinner("Đang tìm câu trả lời..."):
            response = get_gemini_response(prompt)
            st.markdown(response)
    
    # Thêm phản hồi của Gemini vào lịch sử
    st.session_state.messages.append({"role": "assistant", "content": response})

st.markdown("---")
st.markdown("Lưu ý: Chatbot này chỉ trả lời dựa trên các chính sách đã được cung cấp. Nếu câu hỏi nằm ngoài phạm vi, nó sẽ không cung cấp thông tin.")