import os
import pickle
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI
from datasets import load_dataset
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import fitz 
from pathlib import Path
import re
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Few-shot examples for better performance
FEW_SHOT_EXAMPLES = [
    {
        "question": "A 45-year-old man presents with chest pain and shortness of breath. ECG shows ST elevation in leads II, III, and aVF. Which coronary artery is most likely occluded?",
        "options": {"A": "Left anterior descending", "B": "Right coronary artery", "C": "Left circumflex", "D": "Left main coronary artery"},
        "reasoning": "ST elevation in leads II, III, and aVF indicates an inferior wall MI, which is typically caused by RCA occlusion.",
        "answer": "B"
    },
    {
        "question": "A 25-year-old woman has a positive pregnancy test and last menstrual period 8 weeks ago. She reports vaginal bleeding and cramping. Pelvic exam shows a closed cervix. What is the most likely diagnosis?",
        "options": {"A": "Inevitable abortion", "B": "Threatened abortion", "C": "Incomplete abortion", "D": "Missed abortion"},
        "reasoning": "Bleeding with cramping but closed cervix in early pregnancy suggests threatened abortion.",
        "answer": "B"
    }
]


class MedicalRAG:
    def __init__(self, api_key: str, pdf_directory: str = "medical_pdfs", 
                 db_path: str = "medical_vectordb", chunk_size: int = 500):
        """
        Initialize the Medical RAG system
        
        Args:
            api_key: OpenAI API key
            pdf_directory: Directory containing medical PDF files
            db_path: Path to store the vector database
            chunk_size: Size of text chunks for embedding
        """
        self.client = OpenAI(api_key=api_key)
        self.pdf_directory = Path(pdf_directory)
        self.db_path = db_path
        self.chunk_size = chunk_size
        
        # Initialize embedding model (medical domain optimized)
        print("🔄 Loading embedding model...")
        # self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        # Alternative: Use a medical-specific model if available
        self.embedding_model = SentenceTransformer('sentence-transformers/allenai-specter')
        # self.embedding_model = SentenceTransformer('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
        
        self.chroma_client = chromadb.PersistentClient(path=self.db_path)
        self.collection_name = "medical_knowledge"
        
        try:
            self.collection = self.chroma_client.get_collection(self.collection_name)
            print(f"✅ Loaded existing collection with {self.collection.count()} documents")
        except:
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "Medical knowledge base for RAG"}
            )
            print("🆕 Created new collection")

    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from PDF using PyMuPDF for better extraction"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            print(f"❌ Error extracting from {pdf_path}: {e}")
            # fallback to PyPDF2
            try:
                with open(pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text()
                return text
            except Exception as e2:
                print(f"❌ Fallback also failed for {pdf_path}: {e2}")
                return ""

    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        # remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # remove page numbers and headers/footers 
        text = re.sub(r'\n\d+\n', '\n', text)
        # remove short lines 
        lines = text.split('\n')
        lines = [line.strip() for line in lines if len(line.strip()) > 10]
        return '\n'.join(lines)

    def chunk_text(self, text: str, source: str) -> List[Dict[str, Any]]:
        """Split text into chunks for embedding"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < self.chunk_size:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append({
                        'text': current_chunk.strip(),
                        'source': source,
                        'length': len(current_chunk)
                    })
                current_chunk = sentence
        
        # last chunk
        if current_chunk:
            chunks.append({
                'text': current_chunk.strip(),
                'source': source,
                'length': len(current_chunk)
            })
        
        return chunks

    def process_pdfs(self):
        """Process all PDFs in the directory and add to vector database"""
        if not self.pdf_directory.exists():
            print(f"❌ PDF directory {self.pdf_directory} does not exist!")
            print("📁 Please create the directory and add your medical PDF files.")
            return
        
        pdf_files = list(self.pdf_directory.glob("*.pdf"))
        if not pdf_files:
            print(f"❌ No PDF files found in {self.pdf_directory}")
            return
        
        print(f"🔄 Processing {len(pdf_files)} PDF files...")
        
        all_chunks = []
        for pdf_path in pdf_files:
            print(f"📖 Processing: {pdf_path.name}")
            
            text = self.extract_text_from_pdf(pdf_path)
            if not text.strip():
                print(f"⚠️  No text extracted from {pdf_path.name}")
                continue
            
            text = self.clean_text(text)
            
            chunks = self.chunk_text(text, pdf_path.name)
            all_chunks.extend(chunks)
            print(f"  ✅ Created {len(chunks)} chunks")
        
        if not all_chunks:
            print("❌ No text chunks created from PDFs")
            return
        
        print(f"🔄 Creating embeddings for {len(all_chunks)} chunks...")
        
        # embeddings in batches
        batch_size = 100
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i+batch_size]
            
            texts = [chunk['text'] for chunk in batch]
            
            embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
            
            ids = [f"chunk_{i+j}" for j in range(len(batch))]
            metadatas = [{'source': chunk['source'], 'length': chunk['length']} 
                        for chunk in batch]
            
            self.collection.add(
                documents=texts,
                embeddings=embeddings.tolist(),
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"  ✅ Processed batch {i//batch_size + 1}/{(len(all_chunks)-1)//batch_size + 1}")
        
        print(f"🎉 Successfully processed {len(all_chunks)} chunks!")
        print(f"📊 Total documents in collection: {self.collection.count()}")

    def retrieve_relevant_context(self, question: str, top_k: int = 5) -> List[str]:
        """Retrieve relevant context for a question"""
        if self.collection.count() == 0:
            return []
        
        # Create query embedding
        query_embedding = self.embedding_model.encode([question])
        
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k
        )
        
        return results['documents'][0] if results['documents'] else []

    def answer_with_rag(self, question: str, options: Dict[str, str], 
                       system_prompt: str, top_k: int = 3) -> str:
        """Answer a question using RAG with few-shot examples"""
        # Retrieve context
        context_chunks = self.retrieve_relevant_context(question, top_k)
        
        # enhanced prompt with context
        print(f"Retrieved {len(context_chunks)} chunks:")
        context_text = "\n\n".join([f"Context {i+1}: {chunk}" 
        
        for i, chunk in enumerate(context_chunks)])
        
        # Build prompt with few-shot examples
        prompt = "Here are some examples:\n\n"
        
        for i, example in enumerate(FEW_SHOT_EXAMPLES, 1):
            prompt += f"Example {i}:\n"
            prompt += f"Question: {example['question']}\n\nOptions:\n"
            for letter, text in sorted(example['options'].items()):
                prompt += f"{letter}. {text}\n"
            prompt += f"\nThinking: {example['reasoning']}\n"
            prompt += f"Answer: {example['answer']}\n\n"
        
        prompt += "Now answer this question:\n\n"
        
        if context_chunks:
            prompt += f"""Based on the following medical context, answer the question.

CONTEXT:
{context_text}

QUESTION: {question}

OPTIONS:
"""
            for letter, text in sorted(options.items()):
                prompt += f"{letter}. {text}\n"
            
            prompt += "\nThink through this step-by-step, then respond with ONLY the letter of the correct answer (A, B, C, or D)."
        else:
            prompt += f"Question: {question}\n\nOptions:\n"
            for letter, text in sorted(options.items()):
                prompt += f"{letter}. {text}\n"
            prompt += "\nThink through this step-by-step, then respond with ONLY the letter of the correct answer (A, B, C, or D)."
        
        resp = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        raw_response = resp.choices[0].message.content.strip()
        model_ans = None
        for char in raw_response.upper():
            if char in "ABCD":
                model_ans = char
                break
        return model_ans if model_ans else "A"  # Default fallback

def main():

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file.")

    SYSTEM_PROMPT = (
    "You are a medical expert taking the USMLE exam. "
    "Use the provided medical context AND your medical knowledge. "
    "Think step by step: 1) What is the key clinical finding? "
    "2) What condition does this suggest? 3) What's the best answer? "
    "Respond with ONLY the letter (A, B, C, or D)."
    )
    NUM_QUESTIONS = 50
    PDF_DIRECTORY = "medical_pdfs"  # Create this directory and add your PDFs
    
    print("🏥 Initializing Medical RAG System...")
    rag_system = MedicalRAG(
        api_key=api_key,
        pdf_directory=PDF_DIRECTORY,
        db_path="medical_vectordb"
    )
    
    # Uncomment the next line to process PDFs (only needed once or when adding new PDFs)
    # rag_system.process_pdfs()
    
    print("📊 Loading MedQA dataset...")
    dataset = load_dataset("GBaker/MedQA-USMLE-4-options")
    test_data = dataset["test"].select(range(NUM_QUESTIONS))
    
    print(f"🧪 Testing RAG system on {NUM_QUESTIONS} questions...")
    correct = 0
    
    for idx, sample in enumerate(test_data):
        question = sample["question"]
        opts = sample["options"]
        # idx instead of answer
        correct_answer = sample["answer_idx"]
        
        if isinstance(correct_answer, int):
            correct_answer = chr(ord('A') + correct_answer)
        elif isinstance(correct_answer, str):
            correct_answer = correct_answer.upper()
        
        # answer using RAG
        model_ans = rag_system.answer_with_rag(question, opts, SYSTEM_PROMPT)
        
        if model_ans == correct_answer:
            correct += 1
        
        print(f"Q{idx+1}: Model answered '{model_ans}', correct was '{correct_answer}' "
              f"{'✅ Correct!' if model_ans == correct_answer else '❌ Incorrect'}")
    
    print(f"\n🎯 RAG Accuracy on {NUM_QUESTIONS} questions: "
          f"{correct}/{NUM_QUESTIONS} = {correct/NUM_QUESTIONS:.1%}")

if __name__ == "__main__":
    main()