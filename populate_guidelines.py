import chromadb


def populate_database():
    print("Connecting to local ChromaDB...")
    # 连接到与主程序相同的数据库路径
    chroma_client = chromadb.PersistentClient(path="./chroma_db")

    # 获取或创建专门用于存放医学指南的集合
    collection = chroma_client.get_or_create_collection(name="medical_guidelines")

    # 定义标准 AHA/ACC STEMI 临床指南文本 (模拟真实指南文献)
    stemi_guideline_text = """
    AHA/ACC Clinical Practice Guidelines for the Management of Patients with Acute ST-Segment Elevation Myocardial Infarction (STEMI):

    1. Reperfusion Therapy: All patients presenting with STEMI should undergo immediate reperfusion therapy, preferably primary Percutaneous Coronary Intervention (PCI).
    2. Dual Antiplatelet Therapy (DAPT): Patients must be prescribed Aspirin along with a P2Y12 receptor inhibitor (such as Clopidogrel, Ticagrelor, or Prasugrel) to prevent stent thrombosis.
    3. Lipid Management: High-intensity statin therapy (e.g., Atorvastatin 40-80 mg or Rosuvastatin 20-40 mg) should be initiated or continued in all patients with STEMI and no contraindications, regardless of baseline LDL-C levels.
    4. Beta-Blockers: Oral beta-blockers (e.g., Metoprolol, Carvedilol) should be initiated within the first 24 hours in patients with STEMI who do not have signs of heart failure, evidence of a low output state, increased risk for cardiogenic shock, or other contraindications.
    5. ACE Inhibitors: ACE inhibitors (e.g., Lisinopril) should be administered within the first 24 hours to all patients with STEMI with anterior location, heart failure, or ejection fraction less than or equal to 40%, unless contraindicated.
    """

    print("Injecting AHA/ACC STEMI Guidelines into the vector database...")

    # 将指南文本、元数据和唯一 ID 存入 ChromaDB
    collection.add(
        documents=[stemi_guideline_text],
        metadatas=[{"source": "AHA/ACC STEMI Guidelines 2025", "category": "Cardiology"}],
        ids=["guideline_stemi_001"]
    )

    # 验证是否存入成功
    count = collection.count()
    print(f"Injection successful! The 'medical_guidelines' collection now contains {count} document(s).")


if __name__ == "__main__":
    populate_database()