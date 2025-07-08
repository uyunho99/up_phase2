# Chain of Thought 통합 파이프라인
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence, RunnableMap
from langchain_community.vectorstores import FAISS, DistanceStrategy
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import pandas as pd
import os
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.chat_models import AzureChatOpenAI
from langchain.schema.runnable import RunnableLambda
import time
import json
import re
import numpy as np
from langchain.schema import Document
from sklearn.metrics.pairwise import cosine_similarity

def timed(name):
    def wrapper(fn):
        def inner(x):
            print(f"⏱️ [{name}] 시작")
            start = time.time()
            result = fn(x)
            end = time.time()
            print(f"✅ [{name}] 완료 - 소요 시간: {end - start:.2f}초")
            return result
        return inner
    return wrapper

# 환경 설정
os.environ["OPENAI_API_KEY"] = "sk-"


# LLM 설정
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# 데이터 로드
df = pd.read_csv('./all_origin_updated.csv', encoding='utf-8-sig')

# 데이터 로드 및 벡터DB 구축 (기존과 동일)
def setup_vectordb(df):
    corpus_df = df[["ticket_id_hashed", "generated_summary"]].dropna(subset=["generated_summary"])
    
    documents = [
        Document(
            page_content=row["generated_summary"],
            metadata={"ticket_id": row["ticket_id_hashed"], "doc_id": f"doc_{i}"}
        )
        for i, (_, row) in enumerate(corpus_df.iterrows())
    ]
    
    vectordb = FAISS.from_documents(documents, OpenAIEmbeddings())
    return vectordb, documents

# Chain of Thought 통합 프롬프트
cot_prompt = PromptTemplate.from_template("""
<Role>
당신은 제품 피드백을 분석하고 가장 유사한 기존 사례를 찾는 전문가입니다. 
체계적인 사고 과정을 통해 단계별로 분석을 수행하세요.
</Role>

<Task>
주어진 제품 피드백에 대해 다음 사고 과정을 따라 분석하세요:

**입력 정보:**
- 컴포넌트: {components}
- 피드백 내용: {generated_summary}
- 상위 10개 유사 문서: {top_10_documents}

**분석 과정을 단계별로 사고하세요:**

**Step 1: 피드백 유형 판단**
먼저 이 피드백이 어떤 유형인지 판단해보겠습니다.
- 신규 기능 요청이나 개선 제안인가요? → "Proposal"
- 오류, 고장, 결함, 불만, 불편, 불쾌, 단순문의, 의견 등인가요? → "ICC"

피드백 내용을 분석해보니...
[여기서 단계별 사고 과정을 작성]

**판단 결과:** [Proposal 또는 ICC]

**Step 2: 제안사항 분리 (Proposal인 경우만)**
만약 Proposal이라면, 여러 개의 제안이 포함되어 있는지 확인해보겠습니다.
- "and", 구두점, 여러 문장 등을 통해 구분되는 별개의 아이디어가 있는지 살펴봅니다.

분석 결과...
[사고 과정]

**분리된 제안들:** [JSON 배열 형태]

**Step 3: 첫 번째 제안 선택**
여러 제안 중 원문에서 첫 번째로 나타나는 제안을 선택하겠습니다.

**선택된 첫 번째 제안:** [텍스트]

**Step 4: 제안문 표준화**
선택된 제안을 표준화된 형태로 요약하겠습니다.
- "The user suggests"로 시작하는 영어 문장으로 변환
- 컴포넌트 정보를 활용하여 맥락 보강

**표준화된 제안문:** [영어 문장]

**Step 5: 유사도 분석**
상위 10개 문서와의 유사성을 분석하겠습니다.
각 문서에 대해:
- 기능 범주의 일치성
- 요구사항의 유사성  
- 해결방안의 관련성
을 종합적으로 평가합니다.

**Step 6: Self-Consistency 검증 (3회 반복)**
가장 유사한 문서를 선택하기 위해 3번의 독립적인 판단을 수행하겠습니다.

**1번째 판단:**
- 기능 범주 일치성: [0-25점]
- Claim 커버리지: [0-25점] 
- 근거 충실도: [0-25점]
- 설명 흐름 유사성: [0-25점]
- 평가 총점: [0-100점]
- 선택 문서: [ticket_id]
- 신뢰도: [0-100%]

**2번째 판단:**
[같은 형식으로 반복]

**3번째 판단:**
[같은 형식으로 반복]

**Step 7: 최종 통합 판단**
3번의 판단 결과를 종합하여 최종 결론을 도출하겠습니다.

통합 평가 기준:
1. Self-Consistency 평균 신뢰도 (가중치 0.3)
2. 동일 문서 반복 선택 여부 (+10점 보정)
3. Self-Consistency 판단 총점 평균 (가중치 0.2)
4. RAG 기반 Cosine 유사도 (가중치 0.2)
5. Faithfulness (가중치 0.15)
6. Context Recall (가중치 0.15)

**최종 결론:**

> ✅ 최종 추천 문서: [문서의 Ticket ID]
> 📄 선택된 문서 요약: [핵심 내용 요약]
> 🔒 추천 신뢰도 (%): [0~100 사이 수치]
> 📊 RAG 기반 유사도 (%): [유사도 점수]
> 🧠 선택 근거 요약:
> - [구체적인 선택 이유들]
> - [신뢰도 근거]
> - [유사성 분석 결과]

**사고 과정 완료**
""")

# 유사도 검색 함수 (기존과 동일)
def calculate_cosine_similarity(query, documents, embedding_model):
    index = vectordb.index
    stored_vectors = index.reconstruct_n(0, index.ntotal)
    stored_vectors_np = np.array(stored_vectors)

    query_vector = embedding_model.embed_query(query)
    query_vector_np = np.array(query_vector).reshape(1, -1)

    similarities = cosine_similarity(query_vector_np, stored_vectors_np)[0]
    return similarities

def retrieve_context(proposal):
    embedding_model = OpenAIEmbeddings()
    similarities = calculate_cosine_similarity(proposal, documents, embedding_model)
    top_k_indices = similarities.argsort()[::-1][:10]

    formatted_docs = []
    for idx, i in enumerate(top_k_indices, start=1):
        doc = documents[i]
        score = similarities[i]
        ticket_id = doc.metadata.get("ticket_id", "N/A")
        summary = doc.page_content.strip()[:100]
        
        formatted_docs.append(
            f"문서 {idx}: Ticket ID [{ticket_id}], 유사도: {score:.4f}, 내용: {summary}..."
        )
    
    return "\n".join(formatted_docs)

# Chain of Thought 파이프라인
def cot_pipeline(inputs, vectordb, documents):
    """
    Chain of Thought 방식으로 전체 파이프라인을 실행
    """
    print("\n📥 입력 제안문:")
    print(inputs["generated_summary"])
    
    # 1. 초기 유사도 검색 (CoT에서 참조할 문서들)
    print("\n⏱️ [유사도 검색] 시작")
    start_time = time.time()
    
    # 간단한 initial query로 top-10 문서 검색
    top_10_docs = retrieve_context(inputs["generated_summary"])
    
    end_time = time.time()
    print(f"✅ [유사도 검색] 완료 - 소요 시간: {end_time - start_time:.2f}초")
    
    # 2. Chain of Thought 실행
    print("\n⏱️ [Chain of Thought 분석] 시작")
    start_time = time.time()
    
    cot_result = llm.invoke(cot_prompt.format(
        components=inputs["components"],
        generated_summary=inputs["generated_summary"],
        top_10_documents=top_10_docs
    ))
    
    end_time = time.time()
    print(f"✅ [Chain of Thought 분석] 완료 - 소요 시간: {end_time - start_time:.2f}초")
    
    return cot_result.content

# ICC 데이터프레임 업데이트 함수
def update_icc_df(result_text, inputs, df, icc_df):
    """
    CoT 결과에서 ICC 판별시 데이터프레임 업데이트
    """
    # ICC 판별 여부 확인
    if "판단 결과:** ICC" in result_text:
        # ticket_id와 before/after_change를 df에서 찾기
        ticket_row = df[df["generated_summary"] == inputs["generated_summary"]]
        if not ticket_row.empty:
            ticket_id = ticket_row.iloc[0].get("ticket_id_hashed", "")
            before_change = ticket_row.iloc[0].get("before_change", "")
            after_change = ticket_row.iloc[0].get("after_change", "")
        else:
            ticket_id = ""
            before_change = ""
            after_change = ""

        new_row = {
            "ticket_id": ticket_id,
            "components": inputs.get("components", ""),
            "before_change": before_change,
            "after_change": after_change,
            "ICC": "ICC"
        }
        icc_df = pd.concat([icc_df, pd.DataFrame([new_row])], ignore_index=True)
        
        print("👉 판별 결과 : ICC\n <종료>")
        return True, icc_df
    
    return False, icc_df

# 최종 결과 파싱 함수
def parse_final_result(result_text, inputs, df):
    """
    CoT 결과에서 최종 정보 추출
    """
    # 선택된 문서의 ticket_id 추출
    match_ticket = re.search(r"✅ 최종 추천 문서: \[(.*?)\]", result_text)
    ticket_id = match_ticket.group(1) if match_ticket else ""

    # RAGAS 점수 추출
    match_ragas = re.search(r"📊 RAG 기반 유사도.*?:\s*(\d+)", result_text)
    ragas_score = int(match_ragas.group(1)) if match_ragas else 0

    # 선택된 문서 정보 조회
    selected_row = df[df["ticket_id_hashed"] == ticket_id]
    if not selected_row.empty:
        keyword = selected_row.iloc[0].get("keyword", "N/A")
        before_change = selected_row.iloc[0].get("before_change", "")
        after_change = selected_row.iloc[0].get("after_change", "")
    else:
        keyword = "N/A"
        before_change = ""
        after_change = ""

    return {
        "ticket_id": ticket_id,
        "components": inputs["components"],
        "before_change": before_change,
        "after_change": after_change,
        "generated_summary": inputs["generated_summary"],
        "ragas_score": ragas_score,
        "keyword": keyword
    }

# 메인 실행 함수
def main(df, inputs):
    """
    Chain of Thought 방식의 메인 실행 함수
    """
    # 벡터DB 설정
    global vectordb, documents
    vectordb, documents = setup_vectordb(df)
    
    # ICC 데이터프레임 초기화
    icc_df = pd.DataFrame(columns=["ticket_id", "components", "before_change", "after_change", "ICC"])
    
    # CoT 파이프라인 실행
    result_text = cot_pipeline(inputs, vectordb, documents)
    
    # 결과 출력
    print("\n" + "="*50)
    print("📊 Chain of Thought 분석 결과")
    print("="*50)
    print(result_text)
    
    # ICC 체크 및 처리
    is_icc, updated_icc_df = update_icc_df(result_text, inputs, df, icc_df)
    if is_icc:
        return None, updated_icc_df
    
    # Proposal인 경우 최종 결과 파싱
    final_result = parse_final_result(result_text, inputs, df)
    return final_result, updated_icc_df

# 실행 예시
if __name__ == "__main__":
    # 데이터 로드 (실제 경로로 수정 필요)
    # df = pd.read_csv('your_data_path.csv', encoding='utf-8-sig')
    
    # # 입력 예시
    # inputs = {
    #     "components": "로봇청소기",
    #     "generated_summary": "ThinkQ 평면도상에 선을 그어 청소구역을 지정하도록 해주세요"
    # }
    
    # 실행
    result, icc_df = main_cot(df, inputs)