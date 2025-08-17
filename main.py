import os
from dotenv import load_dotenv
import fitz  # PyMuPDF
from pathlib import Path
from typing import Dict, Any, List
import re

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Updated to use current Groq models (not deprecated)
summary_llm = ChatGroq(api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile", temperature=0)
analysis_llm = ChatGroq(api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant", temperature=0.3)
validation_llm = ChatGroq(api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile", temperature=0.2)

def estimate_tokens(text):
    """Rough estimation: 1 token ≈ 4 characters"""
    return len(text) // 4

def chunk_text(text, max_tokens=6000):
    """Split text into chunks that fit within token limits"""
    chunks = []
    sentences = re.split(r'[.!?]+', text)
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
        
        if estimate_tokens(potential_chunk) > max_tokens:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                # Single sentence is too long, split by words
                words = sentence.split()
                word_chunk = ""
                for word in words:
                    if estimate_tokens(word_chunk + " " + word) > max_tokens:
                        if word_chunk:
                            chunks.append(word_chunk.strip())
                            word_chunk = word
                        else:
                            # Single word too long, add as is
                            chunks.append(word)
                    else:
                        word_chunk = word_chunk + " " + word if word_chunk else word
                if word_chunk:
                    chunks.append(word_chunk.strip())
                current_chunk = ""
        else:
            current_chunk = potential_chunk
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using PyMuPDF"""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text()
            if page_text.strip():  # Only add non-empty pages
                text += f"\n\n--- Page {page_num + 1} ---\n\n{page_text}"
        doc.close()
        return text.strip()
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")

def analyze_medical_document_chunk(chunk, chunk_num, total_chunks):
    """Analyze a single chunk of medical document"""
    system_prompt = f"""
You are analyzing part {chunk_num} of {total_chunks} from a medical document. Extract key information and format in markdown:

### Date of Incident
- When did the medical event occur?

### Medical Facility  
- Hospital/clinic name and location

### Healthcare Providers
- Names and roles of medical staff

### Patient Information
- Chief complaints and symptoms
- Vital signs if available
- Relevant medical history

### Medications
- Current medications
- New prescriptions
- Dosages and instructions

### Additional Notes
- Any other relevant medical information

If some sections are not present in this chunk, indicate "Not found in this section".
"""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Analyze this medical document section:\n\n{chunk}")
    ]
    
    try:
        response = analysis_llm.invoke(messages)
        return response.content
    except Exception as e:
        return f"Error analyzing chunk {chunk_num}: {str(e)}"

def analyze_medical_document(text):
    """Analyze medical document, handling long texts by chunking"""
    if estimate_tokens(text) <= 6000:
        # Text is short enough, process normally
        return analyze_medical_document_chunk(text, 1, 1)
    
    # Text is too long, split into chunks
    chunks = chunk_text(text, max_tokens=5000)
    chunk_analyses = []
    
    for i, chunk in enumerate(chunks, 1):
        analysis = analyze_medical_document_chunk(chunk, i, len(chunks))
        chunk_analyses.append(f"## Analysis of Section {i}\n\n{analysis}")
    
    # Combine all chunk analyses
    combined_analysis = "\n\n".join(chunk_analyses)
    
    # Summarize the combined analysis using llama-3.3-70b-versatile
    summary_prompt = """
Combine and consolidate the following medical document analyses into a single comprehensive analysis:

### Date of Incident
### Medical Facility  
### Healthcare Providers
### Patient Information
### Medications
### Additional Notes

Remove duplicates and organize information coherently.
"""
    
    try:
        messages = [
            SystemMessage(content=summary_prompt),
            HumanMessage(content=combined_analysis[:20000])  # Limit input size
        ]
        response = summary_llm.invoke(messages)
        return response.content
    except Exception as e:
        return combined_analysis  # Return combined analysis if summarization fails

def generate_medical_summary(analysis_result):
    """Generate medical summary with token limit handling"""
    # Truncate analysis if too long
    max_analysis_tokens = 8000  # llama-3.3-70b-versatile has larger context
    if estimate_tokens(analysis_result) > max_analysis_tokens:
        # Truncate to fit within limits
        analysis_result = analysis_result[:max_analysis_tokens * 4]
    
    system_prompt = """
Create a concise medical summary in markdown format:

### Key Findings
- Most important medical observations
- Critical symptoms or conditions

### Diagnosis
- Primary diagnosis
- Secondary conditions if any

### Treatment Plan
- Recommended procedures
- Medications prescribed
- Follow-up care instructions

### Additional Notes
- Important considerations
- Special instructions

Keep the summary professional and medically accurate.
"""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Create a medical summary from this analysis:\n\n{analysis_result}")
    ]
    
    try:
        response = summary_llm.invoke(messages)
        return response.content
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def validate_diagnosis(analysis_result, summary):
    """Validate diagnosis with token limit handling"""
    # Truncate inputs if too long
    max_tokens = 6000
    if estimate_tokens(analysis_result) > max_tokens:
        analysis_result = analysis_result[:max_tokens * 4]
    if estimate_tokens(summary) > max_tokens:
        summary = summary[:max_tokens * 4]
    
    system_prompt = """
Provide medical validation assessment in markdown:

### Alignment Analysis
- Do symptoms match the diagnosis?
- Are treatments appropriate for the condition?
- Are prescribed medications suitable?

### Recommendations
- Alternative treatments to consider
- Suggested medication adjustments
- Additional tests that might be beneficial

### Risk Assessment
- Potential complications to monitor
- Drug interaction concerns
- Important follow-up recommendations

Provide objective medical assessment based on standard care practices.
"""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Validate this medical case:\n\nAnalysis:\n{analysis_result}\n\nSummary:\n{summary}")
    ]
    
    try:
        response = validation_llm.invoke(messages)
        return response.content
    except Exception as e:
        return f"Error validating diagnosis: {str(e)}"

def process_medical_document(document_path: str) -> Dict[str, Any]:
    try:
        # Extract text from PDF
        context = extract_text_from_pdf(document_path)
        
        if not context.strip():
            raise ValueError("No text could be extracted from the PDF")
        
        print(f"Extracted text length: {len(context)} characters")
        print(f"Estimated tokens: {estimate_tokens(context)}")
        
        # Analysis
        analysis = analyze_medical_document(context)
        
        # Summary
        summary = generate_medical_summary(analysis)
        
        # Validation
        validation = validate_diagnosis(analysis, summary)
        
        return {
            "analysis": analysis,
            "summary": summary,
            "validation": validation
        }
    except Exception as e:
        raise Exception(f"Error processing document: {str(e)}")
