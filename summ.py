import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import requests

# 요약용 프롬프트 구성 함수
def build_prompt(text: str) -> str:
    return f"""
다음 내용을 간결하고 핵심적으로 요약해 주세요.
- 요약 목적: 전체 맥락과 핵심 정보 파악
- 요약 방식: 중요 개념·원인·결과·결정 사항 중심으로 구조화
- 형식:
    1. 핵심 요약 (3~5문장 이내)
    2. 주요 항목별 정리 (원인, 논의, 해결방안, 결론 등 존재하는 경우)
- 주의: 중복, 불필요한 대화, 장황한 표현은 생략하고 정보 위주로 서술

포맷은 다음과 같이 해주세요:
	- 주제:
	- 논의된 주요 내용:
	- 결정된 사항:
	- 향후 과제 또는 Follow-up:

내용: {text}
"""

API_KEY = "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"

def summarize(text: str, max_new_tokens=1000) -> str:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = build_prompt(text)
    
    # Split text into chunks if it's too long (approximately 4000 tokens)
    max_chunk_length = 9000  # Approximate characters per 4000 tokens
    if len(text) > max_chunk_length:
        chunks = [text[i:i + max_chunk_length] for i in range(0, len(text), max_chunk_length)]
        summaries = []
        
        for chunk in chunks:
            chunk_prompt = build_prompt(chunk)
            data = {
                "model": "google/gemma-3-27b-it:free",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant specialized in summarizing meeting minutes."},
                    {"role": "user", "content": chunk_prompt}
                ]
            }

            response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)

            if response.status_code == 200:
                chunk_summary = response.json()['choices'][0]['message']['content']
                summaries.append(chunk_summary)
            else:
                print(f"Error: {response.status_code}")
                print(response.text)
                return "요약 생성 중 오류가 발생했습니다."
        
        # Combine chunk summaries
        combined_text = "\n\n".join(summaries)
        
        # Final summarization of combined summaries
        data = {
            "model": "google/gemma-3-27b-it:free",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant specialized in summarizing meeting minutes."},
                {"role": "user", "content": build_prompt(combined_text)}
            ]
        }

        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)

        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return "요약 생성 중 오류가 발생했습니다."
    
    else:
        data = {
            "model": "google/gemma-3-27b-it:free",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant specialized in summarizing meeting minutes."},
                {"role": "user", "content": prompt}
            ]
        }

        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)

        if response.status_code == 200:
            summary = response.json()['choices'][0]['message']['content']
            return summary
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return "요약 생성 중 오류가 발생했습니다."
# # 요약 실행 함수
# def summarize(text: str, max_new_tokens=1000) -> str:
#     print('start summarize')
#     print(text)

#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     # 모델과 토크나이저 불러오기
#     model_name = "Bllossom/llama-3.2-Korean-Bllossom-3B"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)

#     prompt = build_prompt(text)
#     input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

#     with torch.no_grad():
#         output_ids = model.generate(
#             input_ids,
#             max_new_tokens=max_new_tokens,
#             do_sample=True,
#             temperature=0.7,
#             top_p=0.9,
#             repetition_penalty=1.1,
#             eos_token_id=tokenizer.eos_token_id
#         )

#     output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
#     summary = output_text.split("회의록:")[-1].strip()
#     return summary