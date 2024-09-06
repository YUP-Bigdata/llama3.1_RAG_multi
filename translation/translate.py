import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import hanja
import re

def translate_model_call():
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct",
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained("LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct")
    return tokenizer, model

def extract_korean_text(text):
    pattern = r"\[\|assistant\|\](.*?)\[\|endofturn\|\]"
    results = re.findall(pattern, text, re.DOTALL)
    return results[0].strip() if results else None

def pdf_translate(model, tokenizer, main_text):
    system = "You are a highly accurate translation model. Translate the following paragraph from Chinese characters and English words into Korean. Provide only one translation without repetition or alternative versions."
    
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": main_text}
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    
    input_ids = input_ids.to('cuda')
    
    with torch.amp.autocast('cuda'):
        output = model.generate(
            input_ids=input_ids,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=1024
        )
    
    result = tokenizer.decode(output[0])
    korean_text = extract_korean_text(result)
    return hanja.translate(korean_text, 'substitution') if korean_text else None
