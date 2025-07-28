import json

# Đọc file credentials JSON
with open("solar-catfish-466509-p0-3b59f69c0484.json", "r") as f:
    data = json.load(f)

# Chuyển thành chuỗi JSON một dòng và escape đúng cách
cred_str = json.dumps(data).replace("\\", "\\\\").replace('"', '\\"')

# Ghi vào cuối file .env
with open(".env", "a", encoding="utf-8") as f:
    f.write(f'GOOGLE_CRED="{cred_str}"\n')

print("✅ Đã ghi GOOGLE_CRED vào file .env")
