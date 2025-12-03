import os
from dotenv import load_dotenv
import google.generativeai as genai

# Fix Windows UTF-8 printing
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Load API key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-2.5-flash")

# Ensure output folder exists
output_folder = "pipelines"
os.makedirs(output_folder, exist_ok=True)

csv_filename = os.path.join(output_folder, "extracted_supports.csv")

prompt = """
You are an expert in interpreting engineering drawings.

Extract all support-related information from this PDF and return it as a CSV table with the following columns:

DRAWING_NAME,SUPPORT_NAME,SUPPORT_ANNOTATION

Rules:
- Assign drawing names D1, D2, ... for pages.
- SUPPORT_NAME includes S<number> style tags.
- SUPPORT_ANNOTATION includes text like P-4P-SPS-14970.
- Output ONLY the CSV table.
"""

response = model.generate_content([
    prompt,
    {
        "mime_type": "application/pdf",
        "data": open("drawings/4103-20-92DJ-1400-031_Bare.pdf", "rb").read()
    }
])

with open(csv_filename, "w", encoding="utf-8") as f:
    f.write(response.text.strip())

print(f"CSV saved at: {csv_filename}")
