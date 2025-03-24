from bs4 import BeautifulSoup

# 读取 HTML 文件
with open("chat.html", "r", encoding="utf-8") as file:
    soup = BeautifulSoup(file, "html.parser")

# 假设聊天记录在 <div class="message"> 之类的标签中
messages = []
for msg in soup.find_all("div", class_="message"):
    sender = msg.find("span", class_="sender").text.strip()
    text = msg.find("span", class_="text").text.strip()
    messages.append({"sender": sender, "text": text})

# 输出前几条看看效果
print(messages[:5])
