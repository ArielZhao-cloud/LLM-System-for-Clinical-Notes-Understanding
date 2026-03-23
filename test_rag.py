import chromadb

# 1. 创建一个 Chroma 客户端 (这里存在内存中，关闭即销毁。以后我们再改成持久化存储)
print("Initializing Chroma local vector database...")
chroma_client = chromadb.Client()

# 2. 创建一个名为 "oncology_guidelines" 的数据集合 (类似于数据库里的表)
collection = chroma_client.create_collection(name="oncology_guidelines")

print("Converting medical guidelines into vectors and storing them in the database (may download a small model on first run)...")

# 3. 录入模拟的肿瘤学医学指南和文献
# 对应论文中将医学知识库向量化并存储的步骤 [cite: 586, 587]
collection.add(
    documents=[
        "For unresectable or metastatic melanoma with a BRAF V600E mutation not suitable for local therapy, targeted therapy with Dabrafenib in combination with Trametinib is recommended.",
        "In metastatic colorectal cancer, if the patient's tumor exhibits high microsatellite instability (MSI-H) or mismatch repair deficiency (dMMR), Pembrolizumab is recommended as a first-line immunotherapy option.",
        "For patients with early breast cancer who are hormone receptor (HR) positive and HER2 negative, at least 5 years of adjuvant endocrine therapy, such as Tamoxifen, is generally recommended after surgery."
    ],
    metadatas=[
        {"source": "OncoKB_BRAF_Guideline", "disease": "Melanoma"},
        {"source": "ESMO_CRC_Guideline", "disease": "Colorectal Cancer"},
        {"source": "ASCO_Breast_Guideline", "disease": "Breast Cancer"}
    ],
    ids=["doc1", "doc2", "doc3"]
)
print("✅ Knowledge base entry complete!\n")

# 4. 模拟大模型根据患者病情发出检索请求 (Query)
# 假设有一个结直肠癌患者的病理报告显示了特定的生物标志物
query_text = "The patient is diagnosed with metastatic colorectal cancer, and genetic testing shows high microsatellite instability (MSI-H). What are the recommended treatment options?"
print(f"🔍 Search Query: {query_text}")

# 5. 在向量数据库中进行相似度搜索，提取最相关的 Top 1 文献 [cite: 637]
results = collection.query(
    query_texts=[query_text],
    n_results=1  # 只返回最相关的一条
)

print("\n🎉 Search results:")
print("-" * 50)
print(f"Matched document content: {results['documents'][0][0]}")
print(f"Document source (Metadata): {results['metadatas'][0][0]['source']}")
print("-" * 50)