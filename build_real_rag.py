import os
import chromadb
from bs4 import BeautifulSoup
import dspy

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CHROMA_TELEMETRY_DISABLED"] = "1"

print("1. Parsing the XML file and extracting paragraphs...")
xml_file_path = "./grobid_outputs/s43018-025-00991-6.grobid.tei.xml"

with open(xml_file_path, "r", encoding="utf-8") as file:
    soup = BeautifulSoup(file, "xml")

body = soup.find("body")
documents = []
metadatas = []
ids = []

if body:
    sections = body.find_all("div")
    chunk_id = 0
    for section in sections:
        header = section.find("head")
        section_title = header.text.strip() if header else "Unknown Section"

        paragraphs = section.find_all("p")
        for p in paragraphs:
            text = p.text.strip()
            # Only keep paragraphs that are reasonably long to avoid noise
            if len(text) > 50:
                documents.append(text)
                metadatas.append({"source": "Nature Cancer Paper", "section": section_title})
                ids.append(f"chunk_{chunk_id}")
                chunk_id += 1

print(f"Extracted {len(documents)} valid paragraphs.")

print("2. Initializing Chroma Vector DB and storing data...")
chroma_client = chromadb.Client()
# Recreate the collection for this specific paper
try:
    chroma_client.delete_collection(name="nature_cancer_paper")
except Exception:
    pass

collection = chroma_client.create_collection(name="nature_cancer_paper")

# Add documents in batches to avoid overwhelming the local embedding model
batch_size = 50
for i in range(0, len(documents), batch_size):
    collection.add(
        documents=documents[i:i + batch_size],
        metadatas=metadatas[i:i + batch_size],
        ids=ids[i:i + batch_size]
    )
print("Data successfully embedded and stored.")

print("3. Initializing Zhipu LLM...")
api_key = "7bc9d8e4b5444b97b5a25e071c478e2c.P7iDRHY4rhaDZYQk"
my_domestic_model = dspy.OpenAI(
    model='glm-4-flash',
    api_key=api_key,
    api_base='https://open.bigmodel.cn/api/paas/v4/',
    model_type='chat',
    max_tokens=1000
)
dspy.settings.configure(lm=my_domestic_model)

print("4. Querying the document...")
# Let's ask a specific question about the paper's methodology
query_text = "What specific vision models or tools did the AI agent use for interpreting visual data like CT or MRI scans?"
print(f"Question: {query_text}")

retrieval_results = collection.query(
    query_texts=[query_text],
    n_results=3  # Retrieve the top 3 most relevant paragraphs
)

# Combine the retrieved paragraphs into one context block
retrieved_context = "\n".join(retrieval_results['documents'][0])


class PaperQARAG(dspy.Signature):
    """Answer the question about the research paper based ONLY on the provided context."""
    context = dspy.InputField(desc="Extracted paragraphs from the paper")
    question = dspy.InputField(desc="Question about the paper")
    answer = dspy.OutputField(desc="Detailed answer based on context")


predictor = dspy.Predict(PaperQARAG)
response = predictor(context=retrieved_context, question=query_text)

print("\n--- Final Answer ---")
print(response.answer)
print("--------------------")
print("Sources used:")
for meta in retrieval_results['metadatas'][0]:
    print(f"- Section: {meta['section']}")