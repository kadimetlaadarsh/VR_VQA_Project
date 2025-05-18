import os
import csv
import pandas as pd
from PIL import Image
import google.generativeai as genai

# === CONFIGURATION ===
API_KEY = "AIzaSyAsY6rEVJnUyW-Sf5SuKK-VJiUNpN8O6AY"
EXCEL_FILE = "merged_output_final/merged_output_final.csv"
BASE_IMAGE_DIR = "abo-images-small/images/small"  # <- update this
NUM_ROWS = 2000
START_ROW = 45000
OUTPUT_CSV = "merged_vqa_dataset.csv"
# === GEMINI SETUP ===
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# === PROMPT TEMPLATE ===
def build_prompt(product_type, item_name, keywords, node_name):
    return f"""
You are given an image of a product and some metadata. Use both the image and metadata to generate 3 short Visual Question Answering (VQA) pairs. Each answer must be a **single word** that is clearly visible in the image.

⚠️ Rules:
- Base your answers only on the image — do **not** use brand names, text labels, or product specs not visible in the image.
- Do **not** ask about the product name, price, usage, or brand identity.
- Avoid repeating the same question phrasing across samples.

The metadata is:
- Product Type: {product_type}
- Item Name: {item_name}
- Keywords: {keywords}
- Category Path: {node_name}

Instructions:
- **Question 1**: Ask a general, visual question about the main object. Examples:
  - "What object is shown?"
  - "What item is visible?"
  - "What is in the image?"
  - Avoid repeating the same question phrasing across samples.
- **Questions 2 and 3**: Ask **descriptive** questions about **visually obvious** features. Avoid repeating the same **question phrasing** across samples. You can choose your own phrasing for each property. These can be about:
  - Dimensions or measurements clearly shown/ labeled in the image**, such as:
    - "how much is the height?" You must mention the value of the height in the answer
    - "how much is the diameter?" You must mention the value of the diameter in the answer
    - "how much is the width?" You must mention the value of the width in the answer
    - or any relevant dimensions/labels given in the image
  - Color: Ask about the color(s) observed in the image. For example, "What color is the object?" or "What is the primary color?"
  - Shape: Ask about the shape of the object(s) in the image. For example, "What is the shape of the object?" or "Is it circular, square, etc.?"
  - Material: Ask about the material the object is made of. For example, "What material is the object made from?" or "What is it made of?" or "Is it made of wood, metal, plastic, etc.?"
  - Quantity: Ask about how many items are visible in the image. For example, "How many objects are in the image?" or "How many items are present?"
  - Transparency: Ask about the transparency of the object(s). For example, "Is it transparent or opaque?" or "Is the object opaque?"
  - Pattern, texture, or other visual features: Ask about any noticeable patterns or textures. For example, "What texture does the object have?" or "Does it have any specific patterns?"
  🌀 You may use **other relevant visual properties** if these categories are not meaningful or are repetitive for the object — just ensure the answer is visible and requires no external knowledge.

📦 Return your output in this exact **CSV format only** — no extra text:
question1,answer1  
question2,answer2  
question3,answer3

⚠️ Important formatting rules:

Each line must contain a question, followed by a comma, followed by a single-word answer. 
NOTE: For only dimension questions, include its numeric value followed by it's unit without any space.

Use this exact format:

question1,answer1  
question2,answer2  
question3,answer3

❌ Do NOT write just the answers.

❌ Do NOT reverse the format.

✅ Each line = one complete "question,answer" pair.
"""

# === MAIN PROCESS ===
def main():
    df = pd.read_csv(
        EXCEL_FILE,
        skiprows=range(1, START_ROW + 1),  # Skip rows 1–10,000 (after header)
        nrows=NUM_ROWS                     # Read next 2000 rows
    )


    results = []

    for idx, row in df.iterrows():
        image_path = os.path.join(BASE_IMAGE_DIR, str(row['path']))
        if not os.path.isfile(image_path):
            print(f"Skipping missing image: {image_path}")
            continue

        try:
            img = Image.open(image_path).convert("RGB")

            prompt = build_prompt(
                row.get("product_type", ""),
                row.get("item_name", ""),
                row.get("item_keywords", ""),
                row.get("node_name", "")
            )

            response = model.generate_content([prompt, img])

            lines = response.text.strip().split("\n")

            # Remove known header or extra lines
            filtered_lines = []
            for line in lines:
                if line.strip().lower().startswith("question") or "csv output" in line.lower():
                    continue
                if ',' in line:
                    filtered_lines.append(line.strip())

            if len(filtered_lines) != 3:
                print(f"Unexpected format for {image_path}, skipping...\nFiltered lines:\n{filtered_lines}")
                continue

            q1, a1 = [s.strip() for s in filtered_lines[0].split(",", 1)]
            q2, a2 = [s.strip() for s in filtered_lines[1].split(",", 1)]
            q3, a3 = [s.strip() for s in filtered_lines[2].split(",", 1)]


            results.append({
                "path": row["path"],
                "q1": q1, "a1": a1,
                "q2": q2, "a2": a2,
                "q3": q3, "a3": a3
            })

            print(f"Processed {idx+1}/{NUM_ROWS}: {image_path}")

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue

    # === SAVE RESULTS ===
    with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["path", "q1", "a1", "q2", "a2", "q3", "a3"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ Saved {len(results)} entries to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
