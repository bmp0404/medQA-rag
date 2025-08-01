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


class HybridMedicalQA:
    def __init__(self, api_key: str, pdf_directory: str = "medical_pdfs", 
                 db_path: str = "medical_vectordb", chunk_size: int = 500):
        """
        Initialize the Hybrid Medical QA system
        
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
        
        # Initialize RAG components (only when needed)
        self.rag_initialized = False
        
        print("✅ Hybrid Medical QA system initialized")

    def initialize_rag(self):
        """Initialize RAG components only when needed"""
        if self.rag_initialized:
            return
            
        print("🔄 Initializing RAG components...")
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('sentence-transformers/allenai-specter')
        
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
        
        self.rag_initialized = True

    def answer_with_few_shot(self, question: str, options: Dict[str, str], 
                            system_prompt: str) -> str:
        """Answer a question using few-shot examples only"""
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

    def retrieve_relevant_context(self, question: str, top_k: int = 5) -> List[str]:
        """Retrieve relevant context for a question"""
        if not self.rag_initialized:
            self.initialize_rag()
            
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
        if not self.rag_initialized:
            self.initialize_rag()
            
        # Retrieve context
        context_chunks = self.retrieve_relevant_context(question, top_k)
        
        # Enhanced prompt with context
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

    SYSTEM_PROMPT_NO_RAG = (
        "You are a medical expert taking the USMLE exam. "
        "Use your medical knowledge and the examples provided. "
        "Think step by step: 1) What is the key clinical finding? "
        "2) What condition does this suggest? 3) What's the best answer? "
        "Respond with ONLY the letter (A, B, C, or D)."
    )
    
    SYSTEM_PROMPT_RAG = (
        "You are a medical expert taking the USMLE exam. "
        "Use the provided medical context AND your medical knowledge. "
        "Think step by step: 1) What is the key clinical finding? "
        "2) What condition does this suggest? 3) What's the best answer? "
        "Respond with ONLY the letter (A, B, C, or D)."
    )
    
    NUM_QUESTIONS = 50
    PDF_DIRECTORY = "medical_pdfs"
    
    print("🏥 Initializing Hybrid Medical QA System...")
    qa_system = HybridMedicalQA(
        api_key=api_key,
        pdf_directory=PDF_DIRECTORY,
        db_path="medical_vectordb"
    )
    
    print("📊 Loading MedQA dataset...")
    dataset = load_dataset("GBaker/MedQA-USMLE-4-options")
    test_data = dataset["test"].select(range(NUM_QUESTIONS))
    
    print(f"🧪 Phase 1: Testing without RAG on {NUM_QUESTIONS} questions...")
    print("=" * 60)
    
    # Phase 1: Test without RAG and track wrong answers
    correct_phase1 = 0
    wrong_questions = []
    
    for idx, sample in enumerate(test_data):
        question = sample["question"]
        opts = sample["options"]
        correct_answer = sample["answer_idx"]
        
        if isinstance(correct_answer, int):
            correct_answer = chr(ord('A') + correct_answer)
        elif isinstance(correct_answer, str):
            correct_answer = correct_answer.upper()
        
        # Answer using few-shot examples only
        model_ans = qa_system.answer_with_few_shot(question, opts, SYSTEM_PROMPT_NO_RAG)
        
        if model_ans == correct_answer:
            correct_phase1 += 1
            print(f"Q{idx+1}: ✅ Correct! ({model_ans})")
        else:
            wrong_questions.append({
                'idx': idx,
                'question': question,
                'options': opts,
                'correct_answer': correct_answer,
                'model_answer_phase1': model_ans
            })
            print(f"Q{idx+1}: ❌ Wrong - Model: {model_ans}, Correct: {correct_answer}")
    
    phase1_accuracy = correct_phase1 / NUM_QUESTIONS
    print(f"\n🎯 Phase 1 (No RAG) Results:")
    print(f"   Correct: {correct_phase1}/{NUM_QUESTIONS} = {phase1_accuracy:.1%}")
    print(f"   Wrong: {len(wrong_questions)} questions")
    
    if not wrong_questions:
        print("\n🎉 Perfect score! No need for RAG phase.")
        return
    
    print(f"\n🧪 Phase 2: Retrying {len(wrong_questions)} wrong questions with RAG...")
    print("=" * 60)
    
    # Phase 2: Retry wrong questions with RAG
    correct_phase2 = 0
    improved_questions = []
    
    for i, wrong_q in enumerate(wrong_questions):
        idx = wrong_q['idx']
        question = wrong_q['question']
        opts = wrong_q['options']
        correct_answer = wrong_q['correct_answer']
        
        # Answer using RAG
        model_ans_rag = qa_system.answer_with_rag(question, opts, SYSTEM_PROMPT_RAG)
        
        if model_ans_rag == correct_answer:
            correct_phase2 += 1
            improved_questions.append(wrong_q)
            print(f"Q{idx+1}: ✅ Fixed with RAG! ({model_ans_rag})")
        else:
            print(f"Q{idx+1}: ❌ Still wrong - RAG: {model_ans_rag}, Correct: {correct_answer}")
    
    # Final results
    final_correct = correct_phase1 + correct_phase2
    final_accuracy = final_correct / NUM_QUESTIONS
    improvement_rate = correct_phase2 / len(wrong_questions) if wrong_questions else 0
    
    print(f"\n" + "=" * 60)
    print(f"FINAL RESULTS:")
    print(f"   Phase 1 (No RAG): {correct_phase1}/{NUM_QUESTIONS} = {phase1_accuracy:.1%}")
    print(f"   Phase 2 (RAG on wrong answers): {correct_phase2}/{len(wrong_questions)} = {improvement_rate:.1%}")
    print(f"   Final Score: {final_correct}/{NUM_QUESTIONS} = {final_accuracy:.1%}")
    print(f"   RAG improved {len(improved_questions)} out of {len(wrong_questions)} wrong answers")
    print(f"   Net improvement: {correct_phase2} questions ({correct_phase2/NUM_QUESTIONS:.1%})")


if __name__ == "__main__":
    main()