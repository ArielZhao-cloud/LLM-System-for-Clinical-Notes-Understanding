import os
import fitz  # PyMuPDF
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==========================================
# 1. 配置参数
# ==========================================
# 请将你要读取的真实医学指南 PDF 放在这个路径下
PDF_PATH = "./data/2013_ACCF_AHA_STEMI_Guidelines.pdf"
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "medical_guidelines"


def extract_text_from_pdf(pdf_path):
    """使用 PyMuPDF 逐页提取 PDF 文本，并保留页码信息"""
    print(f"Loading PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    pages_data = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text").strip()
        if text:
            pages_data.append({
                "page_number": page_num + 1,
                "text": text
            })
    print(f"Successfully extracted {len(pages_data)} pages of text.")
    return pages_data


def chunk_and_vectorize(pages_data):
    """智能切分文本并存入向量数据库"""
    print("Initializing Text Splitter and ChromaDB...")

    # 使用 LangChain 的字符切分器，确保不会把一句话切断
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # 每个文本块大约 800 个字符
        chunk_overlap=150,  # 上下文重叠 150 个字符，防止信息断层
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    # 获取或创建集合（如果之前有旧的测试数据，建议换个名字或者清空）
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    documents = []
    metadatas = []
    ids = []

    chunk_id_counter = 1

    print("Chunking documents...")
    for page in pages_data:
        chunks = text_splitter.split_text(page["text"])
        for chunk in chunks:
            documents.append(chunk)
            # 核心创新：在元数据中注入页码和来源，实现可溯源性
            metadatas.append({
                "source": os.path.basename(PDF_PATH),
                "page": page["page_number"]
            })
            ids.append(f"chunk_{chunk_id_counter}")
            chunk_id_counter += 1

    print(f"Generated {len(documents)} chunks. Injecting into ChromaDB...")

    # 批量存入向量数据库 (ChromaDB 默认会调用轻量级模型自动进行 Embedding)
    # 为了避免内存溢出，分批注入
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        collection.upsert(
            documents=documents[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
            ids=ids[i:i + batch_size]
        )
        print(f"  -> Injected batch {i // batch_size + 1}")

    print("Real-world RAG ingestion complete!")


if __name__ == "__main__":
    if not os.path.exists(PDF_PATH):
        print(f"Error: Please place your PDF file at {PDF_PATH}")
    else:
        pages_data = extract_text_from_pdf(PDF_PATH)
        chunk_and_vectorize(pages_data)