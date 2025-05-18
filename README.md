# Visual Reasoning for E-commerce Product Images

This project focuses on **Visual Question Answering (VQA)** for product images from the Amazon Berkeley Objects (ABO) dataset. It combines metadata and visual data to curate a VQA dataset and evaluates multiple models including BLIP and VILT, with and without LoRA fine-tuning.

---

## 📦 Dataset

We use two main components from the **ABO Dataset**:

- 🔗 [Images + Metadata CSV](https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-images-small.tar): Contains images with associated metadata like width, height, and file path.
- 🔗 [Listings JSON Files](https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-listings.tar): Contains additional metadata such as keywords and product taxonomy (nodes).

---

## 🧹 Data Curation

### 1. **Preprocessing**
- **Script**: `SCRIPTS/data_preprocessing.py`
- **Function**: Merges the metadata from the images CSV and the `listings.json` files using `image_id` as the key.
- **Output**: A curated dataset with metadata merged into a single CSV.

### 2. **VQA Prompt Generation**
- **Notebook**: `SCRIPTS/datacuration.ipynb`
- **Function**:
  - Uses the [Gemini API](https://deepmind.google/discover/blog/google-gemini/) to generate VQA prompts.
  - Combines image metadata and visual features to produce question-answer (QA) pairs.
  - Outputs a final curated VQA dataset.

✅ Example Output:  
[merged_vqa_dataset_output.csv](https://github.com/kadimetlaadarsh/VR_VQA_Project/blob/master/RESULTS/merged_vqa_dataset_output.csv)

---

## 🧠 Baseline Models & Fine-tuning

### 🔹 BLIP (Bootstrapped Language Image Pretraining)
- **Baseline**: `SCRIPTS/BLIP.ipynb`
- **LoRA r=8**: `SCRIPTS/BLIP-LORA-8.ipynb`  
  🔗 [Results](https://github.com/kadimetlaadarsh/VR_VQA_Project/blob/master/RESULTS/LORA_PRED_8_with_bertscore.csv)
- **LoRA r=16**: `SCRIPTS/BLIP-LORA-16.ipynb`  
  🔗 [Results](https://github.com/kadimetlaadarsh/VR_VQA_Project/blob/master/RESULTS/LORA_PRED_16_with_bertscore.csv)

🔗 [BLIP Baseline Results](https://github.com/kadimetlaadarsh/VR_VQA_Project/blob/master/RESULTS/BLIP_results_normalized.csv)

---

### 🔹 VILT (Vision-and-Language Transformer)
- **Baseline**: `SCRIPTS/VILT_baseline.ipynb`  
  🔗 [Results](https://github.com/kadimetlaadarsh/VR_VQA_Project/blob/master/RESULTS/VILT_vqa_baseline_results.csv)
- **LoRA**: `SCRIPTS/VILT_lora.ipynb`

---

## 📊 Evaluation Metrics

### ✅ Implemented in: `SCRIPTS/f1_Hm.py`
We evaluate model performance using:
- **Accuracy**
- **F1 Score (Harmonic Mean)**
- **BERTScore** (included in result CSVs)

---

## 📁 Repository Structure

