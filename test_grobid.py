from grobid_client.grobid_client import GrobidClient
import os

print("Initializing GROBID client...")
# 1. Connect to the local GROBID service running via Docker
client = GrobidClient(config_path=None)
client.config = {
    "grobid_server": "http://localhost:8070",
    "batch_size": 1000,
    "sleep_time": 5,
    "timeout": 60
}

# 2. Specify the directories for input PDFs and output XMLs
input_dir = "./pdfs"
output_dir = "./grobid_outputs"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"Processing PDFs in {input_dir}...")

# 3. Call GROBID to process the full text of the PDF
client.process("processFulltextDocument", input_dir, output_dir, consolidate_citations=False, force=True)

print(f"Processing complete! Check the {output_dir} folder for structured XML files.")