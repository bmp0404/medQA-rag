import os
from typing import Dict
from dotenv import load_dotenv
from openai import OpenAI
from datasets import load_dataset

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


class MedicalQA:
    def __init__(self, api_key: str):
        """
        Initialize the Medical QA system with few-shot examples
        
        Args:
            api_key: OpenAI API key
        """
        self.client = OpenAI(api_key=api_key)
        print("✅ Medical QA system initialized")


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

def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file.")

    SYSTEM_PROMPT = (
    "You are a medical expert taking the USMLE exam. "
    "Use your medical knowledge and the examples provided. "
    "Think step by step: 1) What is the key clinical finding? "
    "2) What condition does this suggest? 3) What's the best answer? "
    "Respond with ONLY the letter (A, B, C, or D)."
    )
    NUM_QUESTIONS = 50
    
    print("🏥 Initializing Medical QA System...")
    qa_system = MedicalQA(api_key=api_key)
    
    print("📊 Loading MedQA dataset...")
    dataset = load_dataset("GBaker/MedQA-USMLE-4-options")
    test_data = dataset["test"].select(range(NUM_QUESTIONS))
    
    print(f"🧪 Testing few-shot system on {NUM_QUESTIONS} questions...")
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
        
        # answer using few-shot examples
        model_ans = qa_system.answer_with_few_shot(question, opts, SYSTEM_PROMPT)
        
        if model_ans == correct_answer:
            correct += 1
        
        print(f"Q{idx+1}: Model answered '{model_ans}', correct was '{correct_answer}' "
              f"{'✅ Correct!' if model_ans == correct_answer else '❌ Incorrect'}")
    
    print(f"\n🎯 Few-shot Accuracy on {NUM_QUESTIONS} questions: "
          f"{correct}/{NUM_QUESTIONS} = {correct/NUM_QUESTIONS:.1%}")

if __name__ == "__main__":
    main()