from bs4 import BeautifulSoup
import os

print("Starting to extract the main text from TEI XML...")

# 1. 设定刚刚生成的 XML 文件路径
# 替换为你的实际文件名（通常是 pdf 文件名加上 .grobid.tei.xml 后缀）
xml_file_path = "./grobid_outputs/s43018-025-00991-6.grobid.tei.xml"

if not os.path.exists(xml_file_path):
    print(f"❌ Cannot find the file: {xml_file_path}")
    exit()

# 2. 读取并解析 XML 文件
with open(xml_file_path, "r", encoding="utf-8") as file:
    soup = BeautifulSoup(file, "xml")

# 3. 定位到文档的正文部分 (<body> 标签)
body = soup.find("body")

if body:
    print("\nSuccessfully located the body text. Extracting content:\n")
    print("=" * 60)

    # 4. 遍历正文中的每一个区块 (<div> 标签)
    sections = body.find_all("div")

    for section in sections:
        # 提取当前区块的标题 (<head> 标签)
        header = section.find("head")
        if header:
            # 打印带编号的标题
            n = header.get('n', '')  # 获取标题编号，比如 "1." 或 "2.1"
            title = header.text.strip()
            print(f"\n[Section: {n} {title}]")

        # 提取该区块下的所有正文段落 (<p> 标签)
        paragraphs = section.find_all("p")
        for p in paragraphs:
            # 去除多余的空格和换行
            clean_text = p.text.strip()
            if clean_text:
                print(f"{clean_text[:150]}... (Total {len(clean_text)} characters)")

    print("\n" + "=" * 60)
    print("Extraction complete!")
else:
    print("Could not find the <body> tag in the XML. The PDF might be empty or scanned.")