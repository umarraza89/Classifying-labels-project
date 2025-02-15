import os
import re
import requests
import pandas as pd
import pdfplumber
from tqdm import tqdm
import signal

# Configuration
PDF_FOLDER = os.path.expanduser("~/Documents/paper/mydocuments")  # Folder containing PDFs
CSV_PATH = os.path.join(PDF_FOLDER, "out.csv")  # Output CSV file path
CATEGORIES = [
    "Graph-Based Learning",
    "Optimization Algorithms",
    "Machine Learning Theory",
    "Reinforcement Learning & Bandits",
    "Applied AI in Healthcare & Web Systems"
]
API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
HEADERS = {"Authorization": f"Bearer {os.environ.get('HF_API_KEY')}"}  # API Key from Hugging Face

# Global flag for interruption
interrupted = False

def signal_handler(sig, frame):
    """Handles Ctrl+C interruption to save progress before exiting"""
    global interrupted
    interrupted = True
    print("\nInterruption received. Saving progress and exiting...")

signal.signal(signal.SIGINT, signal_handler)

def extract_pdf_metadata(pdf_path):
    """Extracts title and abstract from a PDF file"""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            title = None
            abstract = None

            # Extract title: Assume first H1 heading (Largest Font Size)
            first_page = pdf.pages[0]
            text = first_page.extract_text()
            if text:
                lines = text.split("\n")
                title = lines[0].strip()

            # Extract abstract: Search for the keyword "Abstract"
            for page in pdf.pages[:3]:  # Check the first 3 pages for abstract
                text = page.extract_text()
                if not text:
                    continue
                
                abstract_match = re.search(
                    r'(Abstract|ABSTRACT)[\s:]*([\s\S]+?)(?=\n\n|\n1\s|Introduction|INTRODUCTION|$)',
                    text,
                    re.IGNORECASE
                )
                
                if abstract_match:
                    abstract = abstract_match.group(2).strip()
                    break
            
            return title, abstract
    except Exception as e:
        print(f"Error processing {pdf_path}: {str(e)}")
        return None, None

def classify_paper(title, abstract):
    """Classifies a research paper using Hugging Face API"""
    if not os.environ.get("HF_API_KEY"):
        print("ERROR: HF_API_KEY environment variable not set!")
        return "Authentication Error"
    
    try:
        print(f"\nProcessing: {title} ...")
        combined_text = f"{title}. {abstract}"[:2000]  # Limit text length to 2000 characters

        payload = {
            "inputs": combined_text,
            "parameters": {
                "candidate_labels": CATEGORIES,
                "multi_label": False
            }
        }

        response = requests.post(API_URL, headers=HEADERS, json=payload)
        response_data = response.json()

        if response.status_code != 200:
            print(f"API Error: {response_data.get('error', 'Unknown error')}")
            return "API Error"

        if 'error' in response_data:
            print(f"Model Error: {response_data['error']}")
            return "Model Error"

        best_label = response_data['labels'][0]  # Get the top category
        highest_score = response_data['scores'][0]

        print(f"✔ Assigned Label: {best_label} (Confidence: {highest_score:.2f})")
        return best_label  # Always assign the highest category
    
    except Exception as e:
        print(f"\nClassification error: {str(e)}")
        return "Classification Error"

def process_pdfs():
    """Processes PDF files and saves results in CSV"""
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
    
    if not pdf_files:
        print("No PDF files found in the directory.")
        return
    
    data = []
    
    with tqdm(total=len(pdf_files), desc="Processing Papers") as pbar:
        for pdf_file in pdf_files:
            if interrupted:
                break
            
            pdf_path = os.path.join(PDF_FOLDER, pdf_file)
            title, abstract = extract_pdf_metadata(pdf_path)

            if title and abstract:
                label = classify_paper(title, abstract)
            else:
                label = "Missing Metadata"

            data.append([pdf_file, title, abstract, label])
            pbar.update(1)
    
    # Save to CSV
    df = pd.DataFrame(data, columns=["pdf_file", "title", "abstract", "label"])
    df.to_csv(CSV_PATH, index=False)
    print("\nLabeling complete. Results saved to:", CSV_PATH)

if __name__ == "__main__":
    print(f"\n{'='*40}")
    print(f"Research Paper Labeling")
    print(f"Using Hugging Face Inference API")
    print(f"{'='*40}\n")
    
    process_pdfs()
    print("\nLabeling complete. Verify results in the CSV file.")
