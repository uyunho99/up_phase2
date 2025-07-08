import pandas as pd
import os
from openai import OpenAI
from new_keyword_pipeline import setup_vectordb, search_top10_keywords, generate_new_keyword
from cot import main as sim_prompt_main

os.environ["OPENAI_API_KEY"] = "sk-"
client = OpenAI()
df = pd.read_csv('./all_origin_updated.csv', encoding='utf-8-sig')

keyword_vectordb = setup_vectordb("./keyword_list.csv")

input = {
    "components": "로봇청소기",
    "generated_summary": "ThinkQ 평면도상에 선을 그어 청소구역을 지정하도록 해주세요"
}

# sim_prompt_main에서 ICC/Proposal/유사도/키워드 추출
result, icc_df = sim_prompt_main(df, input)

ticket_id = result['ticket_id']
components = result['components']
beforechange = result['before_change']
afterchange = result['after_change']
gen_sum = result['generated_summary']
ragas_score = result['ragas_score']
sim_keyword = result['keyword']
    
print(f"\n\n✅ [최종 결과]\n"
      f"ticket_id: {ticket_id}\n"
      f"components: {components}\n"
      f"beforechange: {beforechange}\n"
      f"afterchange: {afterchange}\n"
      f"gen_sum: {gen_sum}\n"
      f"RAGAS_score: {ragas_score}\n"
      f"sim_keyword: {sim_keyword}")

# Proposal 분기 및 처리
if ragas_score >= 90:
    print(f"[{ticket_id}] 유사도 90↑ 기존 키워드 업로드: {sim_keyword}")
else:
    top10_keywords = search_top10_keywords(keyword_vectordb, gen_sum, top_k=10)
    result = generate_new_keyword(
        client, ticket_id, components, beforechange, afterchange,
        gen_sum, ragas_score, sim_keyword, top10_keywords
    )
    print(f"[{ticket_id}] 유사도 90↓ 새 키워드 생성: {result['new_keyword']}")