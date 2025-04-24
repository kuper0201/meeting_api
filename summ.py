import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 디바이스 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# 모델과 토크나이저 불러오기
model_name = "Bllossom/llama-3.2-Korean-Bllossom-3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)

# 요약용 프롬프트 구성 함수
def build_prompt(text: str) -> str:
    return f"""다음은 회의록입니다. 회의의 주요 주제, 핵심 논의 내용, 결정사항, 그리고 추가 논의가 필요한 사항을 항목별로 정리해서 요약해 주세요.

포맷은 다음과 같이 해주세요:
	- 회의 주제:
	- 논의된 주요 내용:
	- 결정된 사항:
	- 향후 과제 또는 Follow-up:

회의록: {text}
“””

# 요약 실행 함수
def summarize(text: str, max_new_tokens=1000) -> str:
    prompt = build_prompt(text)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id
        )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    summary = output_text.split("요약:")[-1].strip()
    return summary

# 테스트 예시
input_text = """
오늘 회의에서는 신제품 런칭 일정과 관련된 주요 논의가 있었습니다. 
디자인팀은 UI 시안을 다음 주까지 완료하기로 했으며, 마케팅팀은 사전 홍보 전략을 구체화하기로 했습니다. 
기술팀은 현재 베타 테스트 진행 상황을 공유하고, 버그 수정 일정을 보고했습니다.
"""

result = summarize(input_text)
print("\n[요약 결과]")
print(result)
