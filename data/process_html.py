from bs4 import BeautifulSoup
import json

# 读取 HTML 文件内容
file_path = 'chat.html'
with open(file_path, 'r', encoding='utf-8') as file:
    html_content = file.read()

# 解析 HTML
soup = BeautifulSoup(html_content, 'html.parser')

# 查找所有 <script> 标签
script_tags = soup.find_all('script')

messages = []

# 遍历所有 <script> 标签并查找包含 'chatMessages' 的脚本
for script_tag in script_tags:
    if script_tag.string and 'chatMessages' in script_tag.string:
        content = script_tag.string
        # 提取 chatMessages 数组的 JSON 字符串
        start_index = content.find('chatMessages = ') + len('chatMessages = ')
        end_index = content.find('];', start_index) + 1
        chat_messages_str = content[start_index:end_index]

        try:
            # 解析 JSON
            chat_messages = json.loads(chat_messages_str)

            for item in chat_messages:
                if 'text' in item:
                    role = 'girl' if item['is_send'] == 1 else 'boy'
                    messages.append({"role": role, "content": item['text']})

            # 处理合并连续同角色的消息
            merged_messages = []
            for msg in messages:
                if merged_messages and merged_messages[-1]["role"] == msg["role"]:
                    merged_messages[-1]["content"] += " " + msg["content"]
                else:
                    merged_messages.append(msg)
            result = []
            for i in range(len(merged_messages)-1):
                boy = merged_messages[i]['content']
                girl = merged_messages[i+1]['content']
                result.append({"boy": boy, "girl": girl})
            # 保存到 JSON 文件
            with open("chat_dataset.json", "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=4)

            print("✅ 处理完成！合并后的数据已保存到 chat_dataset.json")
        except json.JSONDecodeError as e:
            print(f"❌ JSON 解析错误: {e}")

        break  # 找到 chatMessages 后退出循环
