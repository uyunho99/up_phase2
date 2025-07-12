import pandas as pd
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer
import re
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def setup_vectordb(csv_path: str) -> FAISS:
    """
    CSV 경로를 받아 벡터 DB (FAISS) 구축 후 반환
    """
    df = pd.read_csv(csv_path, encoding='utf-8-sig')

    documents = [
        Document(
            page_content=row["Keyword"],
            metadata={"No": int(row["No"])}
        )
        for _, row in df.iterrows()
    ]

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    keyword_vectordb = FAISS.from_documents(documents, embeddings)

    print(f"✅ 벡터 DB 구축 완료! 저장된 문서 수: {keyword_vectordb.index.ntotal}")
    return keyword_vectordb

def search_top10_keywords(keyword_vectordb, gen_sum: str, top_k: int = 10) -> list:
    """
    벡터 DB에서 gen_sum 기반 Top-k 유사 키워드 검색
    """
    results = keyword_vectordb.similarity_search(gen_sum, k=top_k)
    top10_keywords = []
    print("==== Top-10 유사 키워드 ====")
    for doc in results:
        keyword = doc.page_content
        top10_keywords.append(keyword)
        print(f"- {keyword} (No: {doc.metadata['No']})")
    return top10_keywords



def generate_new_keyword(
    client,
    ticket_id: str,
    components: str,
    beforechange: str,
    afterchange: str,
    gen_sum: str,
    RAGAS_score: int,
    sim_keyword: str,
    top10_keywords: list
    # new_keyword_score: int = 90
) -> dict:
    """
    GPT 호출 -> 새로운 키워드 생성 -> 후처리 -> 결과 dict로 반환
    """

    prompt = f"""
[System]
당신은 고객 요약문(gen_sum)과 벡터 DB의 유사도 Top-10 키워드를 분석하고,
이와 유사하지 않은 새로운 키워드를 하나 생성하는 전문가입니다.

[Input]
- ticket_id: {ticket_id}
- gen_sum: {gen_sum}
- Top-10 유사 키워드: {', '.join(top10_keywords)}

[Instruction]
1) gen_sum의 의미를 잘 반영하고, Top-10 키워드와 중복되지 않는 new_keyword를 하나 생성하세요.
2) new_keyword는 한국어 단일 키워드여야 합니다.
3) 아래 기준으로 평가하세요.
   - 정확성(5점)
   - 포괄성(5점)
   - 간결성(5점)

[생성 키워드 예시]
식품관리
모션감지라이트
외부조명색상변경

[format]
반드시 아래 형식으로 출력하세요:
ticket_id: {ticket_id}
new_keyword: 생성된 하나의 키워드
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    output = response.choices[0].message.content.strip()
    print(f"\n✅ GPT Output:\n{output}")

    match = re.search(r"new_keyword\s*[:：]\s*(.+)", output, re.IGNORECASE)
    if match:
        keyword = match.group(1).strip()
        keyword = keyword.replace("_", " ")
        keyword = re.sub(r"(기능|추가|개선|솔루션)$", "", keyword).strip()
        new_keyword = keyword
    else:
        new_keyword = "ERROR"

    result = {
        "ticket_id": ticket_id,
        "components": components,
        "beforechange": beforechange,
        "afterchange": afterchange,
        "gen_sum": gen_sum,
        "RAGAS_score": RAGAS_score,
        "sim_keyword": sim_keyword,
        "top10_keywords": top10_keywords,
        "new_keyword": new_keyword,
        # "new_keyword_score": new_keyword_score
    }

    print("\n✅ [함수 내부 최종 결과]")
    for k, v in result.items():
        print(f"{k}: {v}")

    return result

# # 실행 예시
# if __name__ == "__main__":
#     import os
#     from openai import OpenAI

#     # ✅ 환경 변수
#     os.environ["OPENAI_API_KEY"] = "sk-"

#     # ✅ 클라이언트
#     client = OpenAI()

#     # ✅ 벡터 DB 구축
#     vectordb = setup_vectordb("./keyword_list.csv")

#     # ✅ gen_sum 예시
#     gen_sum = "The user suggests adding a feature that allows the ice maker to produce ice again when the ice tray is perceived as full but has consumed some ice."

#     # ✅ Top-10 키워드 검색
#     top10_keywords = search_top10_keywords(vectordb, gen_sum, top_k=10)

#     # ✅ 기타 파라미터
#     ticket_id = "64babec"
#     components = "냉장고"
#     beforechange = "이전 변경 내용"
#     afterchange = "이후 변경 내용"
#     RAGAS_score = 95
#     sim_keyword = "아이스메이커기능개선"

#     # ✅ GPT로 새로운 키워드 생성
#     result = generate_new_keyword(
#         client,
#         ticket_id,
#         components,
#         beforechange,
#         afterchange,
#         gen_sum,
#         RAGAS_score,
#         sim_keyword,
#         top10_keywords
#         # new_keyword_score=90
#     )
