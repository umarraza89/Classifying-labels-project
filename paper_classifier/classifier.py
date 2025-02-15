import os
import re
import pandas as pd
import pdfplumber
import torch
from transformers import pipeline
from tqdm import tqdm
import signal

# Configuration
PDF_FOLDER = os.path.expanduser("~/Documents/paper_classifier/mydocuments")
CSV_PATH = os.path.join(PDF_FOLDER, "out.csv")
CATEGORIES = [
    "Graph-Based Learning",
    "Optimization Algorithms",
    "Machine Learning Theory",
    "Reinforcement Learning & Bandits",
    "Applied AI in Healthcare & Web Systems"
]

# Global flag for interruption
interrupted = False

def signal_handler(sig, frame):
    global interrupted
    interrupted = True
    print("\nInterruption received. Saving progress and exiting...")

signal.signal(signal.SIGINT, signal_handler)

# Initialize classifier
model_name = "facebook/bart-large-mnli"
classifier = pipeline(
    "zero-shot-classification",
    model=model_name,
    device=0 if torch.cuda.is_available() else -1
)

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
                title = lines[0].strip()  # Take first heading

            # Extract abstract: Search for the keyword "Abstract"
            for page in pdf.pages[:3]:  # Search in first 3 pages
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
                    break  # Stop after finding the abstract
            
            return title, abstract
    except Exception as e:
        print(f"Error processing {pdf_path}: {str(e)}")
        return None, None

def classify_paper(title, abstract):
    """Classifies the paper and returns the label"""
    if not title or not abstract:
        return "Missing Metadata"
    
    try:
        print(f"\nProcessing: {title} ...")

        combined_text = f"{title}. {abstract}"[:2000]
        
        results = classifier(
            combined_text,
            CATEGORIES,
            multi_label=False,
            hypothesis_template="This paper discusses {}."
        )
        
        highest_score = results['scores'][0]
        best_label = results['labels'][0]

        if highest_score > 0.3:
            print(f"✔ Assigned Label: {best_label} (Confidence: {highest_score:.2f})")
            return best_label
        
        print(f"⚠ Low confidence ({highest_score:.2f}). Assigning 'Low Confidence'.")
        return "Low Confidence"
        
    except Exception as e:
        print(f"\nClassification error: {str(e)}")
        return "Classification Error"

def process_pdfs():
    """Processes PDF files, extracts metadata, classifies them, and saves results in CSV"""
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
    print(f"Model: {model_name}")
    print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print(f"{'='*40}\n")
    
    process_pdfs()
    print("\nLabeling complete. Verify results in the CSV file.")