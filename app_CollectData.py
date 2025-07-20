import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import os
import json
from dotenv import load_dotenv

# --- Load biến môi trường từ file .env ---
load_dotenv()

# --- Đọc Google Sheet credentials từ file JSON ---
try:
    with open("solar-catfish-466509-p0-3b59f69c0484.json", "r") as f:
        creds_dict = json.load(f)
    st.write("✅ Đã đọc file JSON credentials.")
except Exception as e:
    st.error(f"❌ Lỗi đọc file credentials: {e}")
    st.stop()

# --- Lấy Sheet ID từ biến môi trường ---
sheet_id = os.getenv("GOOGLE_SHEET_ID")
if not sheet_id:
    st.error("❌ Thiếu biến môi trường GOOGLE_SHEET_ID. Kiểm tra file .env.")
    st.stop()

# --- Kết nối Google Sheet ---
try:
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)
    sheet = client.open_by_key(sheet_id).sheet1
    st.write("✅ Kết nối Google Sheet thành công.")
except Exception as e:
    import traceback
    st.error("❌ Không kết nối được Google Sheet:")
    st.text(traceback.format_exc())  # <-- log chi tiết lỗi
    st.stop()


# --- UI Chatbox ---
st.title("💬 Chatbox thu thập thông tin")

user_input = st.text_input("Nhập nội dung chat:")

if st.button("Gửi"):
    if user_input.strip():
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            sheet.append_row([now, user_input])
            st.success("✅ Đã lưu vào Google Sheet.")
        except Exception as e:
            st.error(f"❌ Lỗi khi ghi vào Google Sheet: {e}")
    else:
        st.warning("⚠️ Bạn chưa nhập nội dung.")
