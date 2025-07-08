# In[1]:
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence, RunnableMap
from langchain_community.vectorstores import FAISS, DistanceStrategy
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import pandas as pd
import os
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.chat_models import AzureChatOpenAI
from langchain.schema.runnable import RunnableLambda
import time
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

# In[2]:
os.environ["OPENAI_API_KEY"] = "sk-"

# In[3]:
df = pd.read_csv('./all_origin_updated.csv', encoding='utf-8-sig')
# In[4]:
df.head()

# %% 새로운 ICC 데이터프레임 생성
icc_df = pd.DataFrame(columns=["ticket_id", "components", "before_change", "after_change", "ICC"])

# In[5]:
# 환경 구성
llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

#%%
df
# %%In [6]:
# 데이터 준비 및 벡터 DB 구축
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document

# %%dropna() 이후에도 ticket_id가 함께 유지되도록
corpus_df = df[["ticket_id_hashed", "generated_summary"]].dropna(subset=["generated_summary"])

# %%각 문서를 Document 객체로 변환할 때 ticket_id도 함께 metadata로 추가
documents = [
    Document(
        page_content=row["generated_summary"],
        metadata={"ticket_id": row["ticket_id_hashed"], "doc_id": f"doc_{i}"}
    )
    for i, (_, row) in enumerate(corpus_df.iterrows())
]
## Build vector database manually using OpenAIEmbeddings and numpy
# vectordb = FAISS.from_documents(documents, OpenAIEmbeddings(), distance_strategy=DistanceStrategy.COSINE)

#%%
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# %%
vectordb = FAISS.from_documents(documents, OpenAIEmbeddings())

# In[6]:
# 프롬프트 정의

### Step 1: ICC 판단
icc_prompt = PromptTemplate.from_template("""<Role> 당신은 제품 피드백을 읽고 분류하는 전문가입니다.

<Task>: 제품 피드백이 신규 기능 요청이나 개선 제안이면 "Proposal"로 판별합니다.  
그렇지 않고 피드백 내용이 아래의 유형에 해당하면 ICC로 판별합니다.  
ICC란 Issues(오류, 고장, 결함), Complaints(불만, 불편, 불쾌), Comments(단순문의, 의견)로 구성된 일반적인 문제 제기나 의견을 의미합니다.

판별 기준:
- 피드백이 신규 기능 요청 또는 개선 제안이면: "Proposal"
- 피드백에 아래의 단어가 포함되거나 그 의미에 해당하면: "ICC로 판별된 것으로 보여집니다"
  - 오류
  - 고장
  - 결함
  - 불만
  - 불편
  - 불쾌
  - 단순문의
  - 의견

출력 형식:
- Proposal로 판별되면: "Proposal"
- ICC로 판별되면: "ICC"

다른 설명이나 부가적인 텍스트는 출력하지 마세요.

Input Summary:
{generated_summary}
""")


### Step 2: 제안 분리
split_prompt = PromptTemplate.from_template("""<Role3> You extract and organize multiple user proposals from text. </Role3>
<Task>: Your task is to separate each distinct proposal clearly. Identify and number each distinct proposal in the user’s summarized text.

Reasoning: Use a Thought step to identify if the input contains more than one distinct idea (look for keywords like "and", punctuation, or multiple sentences indicating separate ideas). Then Act by listing each proposal separately in the required format. Keep each proposal phrased as a standalone improvement point.

Text:
{generated_summary}

<Format>: If there are multiple proposals in the input, output them as a JSON array of strings, where each string is one proposal. (e.g., ["Proposal 1", "Proposal 2"]). Ensure the JSON is valid. If there is only one proposal, still output a single-element JSON array with that proposal. Do not add any explanatory text, just the JSON.""")

### Step 3: 첫 제안 선택
first_prompt = PromptTemplate.from_template("""<Role4> You select the first proposal based on order of appearance. </Role4>
<Task>: Identify and return the first proposal in the list, exactly as it is, without any additional text.

Reasoning: Use a Thought step to parse the input list and find the first proposal in the original text. Then Act by outputting that first proposal verbatim. Do not include list markers, numbering, or any other proposals.

Original Text: 
{generated_summary}  

List of Proposals:
{proposals}

<Format>: Output the first proposal as a plain text string (no JSON, no list formatting, no quotes around it).""")

### Step 4: 제안문 generated_summary
summary_prompt = PromptTemplate.from_template("""
<Role> You are a home appliance expert, specialized at summarizing electronic product suggestions. </Role>

<Task> Let’s think step by step.

1. Understand the `{proposals}` thoroughly.

2. If the suggestion is too short to summarize or lacks context, instead of asking for more context, generate the summary with the original suggestion itself. You can also use the component `{components}`.

Strictly follow the format below. </Task>

<Format> Please follow these strict output rules:

- Return English text.
- The sentence should start with "The user suggests."
- Output must be in plain string format only.
- Do not include any extra explanations(such as Component:, only print the summary), metadata, notes, or formatting. </Format>
""")

### Step 5: 유사도 정리 (Not calling LLM)

### Step 6: Self-Consistency 판단 (구조화된 평가 기준)
sc_prompt = PromptTemplate.from_template("""
<Role6> 당신은 제안 문장과 유사도 상위 10개의 문서를 바탕으로 가장 유사한 문서를 독립적으로 판단하는 전문가입니다. 이 과정을 총 3회 반복하여 Self-Consistency를 확보합니다. </Role6>

<Task>: 다음 정보를 바탕으로 가장 유사한 문서 하나를 선택하세요. 그리고 아래 항목별 점수를 반드시 숫자(정수)로 평가하고, 마지막에 판단 총점과 사고 신뢰도도 계산해 주세요.

[입력 정보]
Proposal:
{proposal_summary}

Top 10 Documents:
{top_10_table}

<평가 항목>: 아래 각 항목에 대해 정량 점수로 평가하세요.

- 선택 문서 (Tickect_id): [예: fdff64d]
- 기능 범주 일치성 (Context Entity Recall): [0~25점]
- Claim 커버리지 (Context Recall): [0~25점]
- 근거 충실도 (Faithfulness): [0~25점]
- 설명 흐름 유사성: [0~25점]

<정성 평가 항목>: 판단 과정을 스스로 평가하세요.

- 기준 적용 명확성: [0~30점]
- 논리성: [0~30점]
- 설명의 설득력: [0~20점]
- 모호성 없음: [0~20점]

- 평가 총점 합계: [위 네 항목 합계, 0~100점]
- 사고 신뢰도(Self-Eval): [정성 평가 총점에 따라 백분율 변환]

<출력 형식>: 아래 형식을 그대로 따르세요.

[1번째 응답]
- 선택 문서: fdff61d
- 기능 범주 일치성 (Context Entity Recall): 25점
- Claim 커버리지: 20점
- 근거 충실도 (Faithfulness): 24점
- 설명 흐름 유사성: 22점
- 평가 총점 합계: 91점
- 기준 적용 명확성: 30점
- 논리성: 28점
- 설명의 설득력: 18점
- 모호성 없음: 20점
- 사고 신뢰도(Self-Eval): 96%
""")

### Step 7: 최종 판단
choose_prompt = PromptTemplate.from_template("""<Role7> 당신은 반복된 세 번의 판단 결과를 통합하고, 가장 신뢰도 높은 최종 추천 문서를 결정하는 전문가입니다. </Role7>

<Task>: 세 번의 Self-Consistency 기반 판단 응답과 RAG 기반 유사도 점수, RAGAS 기반 정량 지표를 함께 비교 분석하여 최종적으로 가장 유사한 문서를 하나 선정하고, 그 이유를 정리하세요.

Reasoning: 다음 지표들을 모두 수집하여 통합 점수를 계산한 뒤, 가장 종합적으로 유사한 문서를 선택하세요.

<통합 평가 기준>
1. Self-Consistency 평균 신뢰도 (가중치 0.3)
2. 동일 문서 반복 선택 여부 (+10점 보정)
3. Self-Consistency 판단 총점 평균 (가중치 0.2)
4. RAG 기반 Cosine 유사도 (가중치 0.2)
5. Faithfulness (RAGAS 지표) (가중치 0.15)
6. Context Recall (RAGAS 지표) (가중치 0.15)

<최종 선택 기준>
- 종합 점수가 가장 높은 문서를 선택하세요.
- 단, 동일 점수일 경우 Self-Consistency에서 더 자주 선택된 문서를 우선 고려하세요.
- 문서 요약, 기능 목적, 설명 방식이 제안 문장과 가장 유사한지를 근거로 삼아야 합니다.

Self-Consistency 응답 목록:
{self_consistency_responses}

RAG 유사도 목록:
{top_10_table}

<Format>: 아래 형식을 따라 출력하세요.

> ✅ 최종 추천 문서: [문서의 Ticket ID (예: d25d98b)]
> 📄 선택된 문서 요약: [선택된 문서의 핵심 내용 요약]
> 🔒 추천 신뢰도 (%): [0~100 사이 수치, 반복 응답과 신뢰도 평균에 기반]
> 📊 RAG 기반 유사도 (%): [선택된 문서의 RAG 유사도, 없으면 "N/A"]
> 🧠 선택 근거 요약:
> - Self-Consistency 응답 중 동일 문서 반복 선택됨 (+10점)
> - 평균 신뢰도 및 평가 총점이 가장 높음
> - RAG 유사도 및 Faithfulness/Context Recall 점수 또한 우수
> - 제안 문장과의 기능 목적 및 서술 방식이 가장 유사함
""")


# In[10]: inputs 정의

# inputs = {
#     "components": "로봇청소기",
#     "generated_summary": "ThinkQ 평면도상에 선을 그어 청소구역을 지정하도록 해주세요"
# }

# In[11]:
from langchain.schema.runnable import RunnableSequence, RunnableMap, RunnableLambda
import json
import re

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

    rows = []
    docs = []
    result_records = []
    for idx, i in enumerate(top_k_indices, start=1):
        doc = documents[i]
        score = similarities[i]
        ticket_id = doc.metadata.get("ticket_id", "N/A")
        summary = doc.page_content.strip()[:100]
        rows.append(f"| {idx} | {ticket_id} | {score:.4f} | {summary} | |")
        result_records.append({
            "순위": idx,
            "ticket_id": ticket_id,
            "Cosine 유사도": round(score, 4),
            "문서 요약": summary
        })
        docs.append(doc)

    table = "\n".join(["| 순위 | Ticket ID | Cosine 유사도 | 문서 요약 | 주요 키워드 |", "| --- | --- | --- | --- | --- |"] + rows)
    return table, docs, result_records

# Helper: format top-10 Document objects as markdown table for SC prompt
def format_top10_for_prompt(docs):
    rows = ["| 순위 | Ticket ID | Cosine 유사도 | 문서 요약 | 주요 키워드 |", "| --- | --- | --- | --- | --- |"]
    for idx, doc in enumerate(docs, start=1):
        rows.append(f"| {idx} | {doc.metadata.get('ticket_id', 'N/A')} | N/A | {doc.page_content.strip()[:30]}... | |")
    return "\n".join(rows)

def parse_self_consistency_response(text):
    result = {}

    # 선택 문서
    match_doc = re.search(r"- 선택 문서:\s*(\S+)", text)
    result["선택 문서"] = match_doc.group(1) if match_doc else ""

    # 기능 범주 일치성 (Context Entity Recall)
    match_func = re.search(r"- 기능 범주 일치성.*?:\s*(\d+)점", text)
    result["기능 범주 일치성 (Context Entity Recall)"] = int(match_func.group(1)) if match_func else 0

    # Claim 커버리지
    match_claim = re.search(r"- Claim 커버리지:\s*(\d+)점", text)
    result["Claim 커버리지"] = int(match_claim.group(1)) if match_claim else 0

    # 근거 충실도 (Faithfulness)
    match_evidence = re.search(r"- 근거 충실도.*?:\s*(\d+)점", text)
    result["근거 충실도 (Faithfulness)"] = int(match_evidence.group(1)) if match_evidence else 0

    # 설명 흐름 유사성
    match_flow = re.search(r"- 설명 흐름 유사성:\s*(\d+)점", text)
    result["설명 흐름 유사성"] = int(match_flow.group(1)) if match_flow else 0

    # 평가 총점 합계
    match_total = re.search(r"- 평가 총점 합계:\s*(\d+)점", text)
    result["평가 총점 합계"] = int(match_total.group(1)) if match_total else 0

    # 기준 적용 명확성
    match_criteria = re.search(r"- 기준 적용 명확성:\s*(\d+)점", text)
    result["기준 적용 명확성"] = int(match_criteria.group(1)) if match_criteria else 0

    # 논리성
    match_logic = re.search(r"- 논리성:\s*(\d+)점", text)
    result["논리성"] = int(match_logic.group(1)) if match_logic else 0

    # 설명의 설득력
    match_persuasion = re.search(r"- 설명의 설득력:\s*(\d+)점", text)
    result["설명의 설득력"] = int(match_persuasion.group(1)) if match_persuasion else 0

    # 모호성 없음
    match_clarity = re.search(r"- 모호성 없음:\s*(\d+)점", text)
    result["모호성 없음"] = int(match_clarity.group(1)) if match_clarity else 0

    # 사고 신뢰도 (Self-Eval)
    match_self_eval = re.search(r"- 사고 신뢰도\(Self-Eval\):\s*(\d+)%", text)
    result["사고 신뢰도"] = int(match_self_eval.group(1)) if match_self_eval else 0

    return result

def format_self_consistency_table(responses):
    headers = ["평가 항목", "1번째 응답", "2번째 응답", "3번째 응답"]
    rows = [
        ["선택 문서"] + [resp.get("선택 문서", "") for resp in responses],
        ["기능 범주 일치성 (Context Entity Recall)"] + [resp.get("기능 범주 일치성 (Context Entity Recall)", 0) for resp in responses],
        ["Claim 커버리지"] + [resp.get("Claim 커버리지", 0) for resp in responses],
        ["근거 충실도 (Faithfulness)"] + [resp.get("근거 충실도 (Faithfulness)", 0) for resp in responses],
        ["설명 흐름 유사성"] + [resp.get("설명 흐름 유사성", 0) for resp in responses],
        ["평가 총점 합계"] + [resp.get("평가 총점 합계", 0) for resp in responses],
        ["기준 적용 명확성"] + [resp.get("기준 적용 명확성", 0) for resp in responses],
        ["논리성"] + [resp.get("논리성", 0) for resp in responses],
        ["설명의 설득력"] + [resp.get("설명의 설득력", 0) for resp in responses],
        ["모호성 없음"] + [resp.get("모호성 없음", 0) for resp in responses],
        ["사고 신뢰도"] + [resp.get("사고 신뢰도", 0) for resp in responses],
    ]

    # Build markdown table string
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    row_lines = []
    for row in rows:
        row_lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join([header_line, separator_line] + row_lines)

def prepare_self_consistency_input(x):
    return {
        "proposal": x["first_proposal"],
        "top_10_table": x["top_10_table"]
    }
load_context = RunnableLambda(lambda _: {
    "components": inputs["components"],
    "generated_summary": inputs["generated_summary"]
})

def update_icc_df(x):
    if x.get("icc_check") != "ICC":
        return x  # Proposal이면 아무 것도 안 하고 반환

    # ticket_id와 before/after_change를 df에서 찾아오기
    ticket_row = df[df["generated_summary"] == x.get("generated_summary")]
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
        "components": x.get("components", ""),
        "before_change": before_change,
        "after_change": after_change,
        "ICC": "ICC"
    }
    global icc_df
    icc_df = pd.concat([icc_df, pd.DataFrame([new_row])], ignore_index=True)

    return x

chain = RunnableSequence(
    RunnableLambda(lambda x: x),  # 입력값 그대로 흘림

    # Step 1: ICC 판단 (timed)
    RunnableMap({
        "icc_check": RunnableLambda(
            timed("ICC 판단")(
                lambda x: (icc_prompt | llm).invoke({"generated_summary": x["generated_summary"]}).content
            )
        ),
        "components": lambda x: x["components"],
        "generated_summary": lambda x: x["generated_summary"],
    }),

    # ICC 라벨이면 저장 후 종료, Proposal은 다음 단계 진행
    RunnableLambda(lambda x: (
        (update_icc_df(x) or print("👉 판별 결과 : ICC\n <종료>") or exit(0))
    ) if x["icc_check"] == "ICC" else (
        (print("👉 판별 결과 : Proposal") or x)
    )),

    # Step 2: 제안 분리
    RunnableMap({
        "proposals": RunnableLambda(
            timed("제안 분리")(
                lambda x: (split_prompt | llm).invoke({"generated_summary": x["generated_summary"]}).content
            )
        ),
        "components": lambda x: x["components"],
        "generated_summary": lambda x: x["generated_summary"],
    }),

    # Step 3: 첫 제안 선택
    RunnableMap({
        "first_proposal": RunnableLambda(
            timed("첫 제안 선택")(
                lambda x: (first_prompt | llm).invoke({
                    "generated_summary": x["generated_summary"],
                    "proposals": json.dumps(x["proposals"])
                }).content
            )
        ),
        "generated_summary": lambda x: x["generated_summary"],
        "proposals": lambda x: x["proposals"].content if hasattr(x["proposals"], "content") else x["proposals"],
        "components": lambda x: x["components"],
    }),

    # Step 4: 제안문 요약
    RunnableMap({
        "proposal_summary": RunnableLambda(
            timed("제안문 요약")(
                lambda x: llm.invoke(summary_prompt.format(
                    proposals=x["first_proposal"],
                    components=x["components"]
                )).content
            )
        ),
        "first_proposal": lambda x: x["first_proposal"],
        "components": lambda x: x["components"],
        "generated_summary": lambda x: x["generated_summary"],
    }),

    # Step 5: 유사도 Top-10 검색
    RunnableMap({
        "top_10_table": RunnableLambda(
            timed("유사도 Top-10 검색")(
                lambda x: retrieve_context(x["proposal_summary"])[0]
            )
        ),
        "top_10_docs": lambda x: retrieve_context(x["proposal_summary"])[1],
        "top_10_records": lambda x: retrieve_context(x["proposal_summary"])[2],
        "first_proposal": lambda x: x["first_proposal"],
        "proposal_summary": lambda x: x["proposal_summary"],
        "components": lambda x: x["components"],
    }),

    # Step 6: Self-consistency 판단 3회
    RunnableMap({
        "self_consistency_responses": RunnableLambda(
            timed("Self-Consistency 판단 (3회 반복)")(
                lambda x: [
                    llm.invoke(sc_prompt.format(
                        proposal_summary=x["proposal_summary"],
                        top_10_table=format_top10_for_prompt(x["top_10_docs"])
                    )) for _ in range(3)
                ]
            )
        ),
        "first_proposal": lambda x: x["first_proposal"],
        "top_10_table": lambda x: x["top_10_table"],
        "top_10_docs": lambda x: x["top_10_docs"],
        "top_10_records": lambda x: x["top_10_records"],
        "proposal_summary": lambda x: x["proposal_summary"],
        "components": lambda x: x["components"],
    }),

    # Parse and format self_consistency_responses
    RunnableMap({
        "parsed_self_consistency": RunnableLambda(lambda x: [
            parse_self_consistency_response(resp.content if hasattr(resp, "content") else str(resp))
            for resp in x["self_consistency_responses"]
        ]),
        "self_consistency_responses": lambda x: x["self_consistency_responses"],
        "top_10_table": lambda x: x["top_10_table"],
        "first_proposal": lambda x: x["first_proposal"],
        "top_10_docs": lambda x: x["top_10_docs"],
        "top_10_records": lambda x: x["top_10_records"],
        "components": lambda x: x["components"],
    }),

    RunnableMap({
        "self_consistency_table": RunnableLambda(lambda x: format_self_consistency_table(x["parsed_self_consistency"])),
        "self_consistency_responses": lambda x: x["self_consistency_responses"],
        "top_10_table": lambda x: x["top_10_table"],
        "first_proposal": lambda x: x["first_proposal"],
        "top_10_docs": lambda x: x["top_10_docs"],
        "top_10_records": lambda x: x["top_10_records"],
        "components": lambda x: x["components"],
    }),

    # Step 7: 최종 판단
    RunnableMap({
        "final_result": RunnableLambda(
            timed("최종 판단")(
                lambda x: (choose_prompt | llm).invoke({
                    "self_consistency_responses": json.dumps(
                        [resp.content if hasattr(resp, "content") else str(resp) for resp in x["self_consistency_responses"]]
                    ),
                    "top_10_table": x["top_10_table"]
                }).content
            )
        ),
        "self_consistency_responses": lambda x: x["self_consistency_responses"],
        "top_10_table": lambda x: x["top_10_table"],
        "self_consistency_table": lambda x: x["self_consistency_table"],
        "top_10_records": lambda x: x["top_10_records"],
        "components": lambda x: x["components"],
    })
)

# 모듈화: main() 함수 정의 및 반환값 구성
def main(inputs):
    print("\n📥 입력 제안문:")
    print(inputs["generated_summary"])

    result = chain.invoke(inputs)
    print(result["final_result"])

    # RAGAS 점수 추출
    import re
    final_text = result["final_result"]
    match_ragas = re.search(r"📊 RAG 기반 유사도.*?:\s*(\d+)", final_text)
    ragas_score = int(match_ragas.group(1)) if match_ragas else 0

    # 선택된 문서의 ticket_id 추출
    match_ticket = re.search(r"✅ 최종 추천 문서: \[(.*?)\]", final_text)
    ticket_id = match_ticket.group(1) if match_ticket else ""

    selected_row = df[df["ticket_id_hashed"] == ticket_id]
    if not selected_row.empty:
        keyword = selected_row.iloc[0].get("keyword", "N/A")
        before_change = selected_row.iloc[0].get("before_change", "")
        after_change = selected_row.iloc[0].get("after_change", "")
    else:
        keyword = "N/A"
        before_change = ""
        after_change = ""

    components = inputs["components"]
    generated_summary = inputs["generated_summary"]

    return ticket_id, components, before_change, after_change, generated_summary, ragas_score, keyword

if __name__ == "__main__":
    main()
