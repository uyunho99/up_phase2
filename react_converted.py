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

timing_results = []

def timed(name):
    def wrapper(fn):
        def inner(x):
            print(f"⏱️ [{name}] 시작")
            start = time.time()
            result = fn(x)
            end = time.time()
            duration = end - start
            print(f"✅ [{name}] 완료 - 소요 시간: {end - start:.2f}초")
            global timing_results
            timing_results.append((name, duration))
            return result
        return inner
    return wrapper

# In[2]:
os.environ["OPENAI_API_KEY"] = "sk-"

# In[3]:
df = pd.read_csv('./all_origin_updated_sam.csv', encoding='utf-8-sig')
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
original_vectordb = FAISS.from_documents(documents, OpenAIEmbeddings())

# In[6]:
# 프롬프트 정의

### Step 1: ICC Classification
icc_prompt = PromptTemplate.from_template("""<Role> You are an expert in reading and classifying product feedback.

<Task>: If the product feedback contains new feature requests or improvement suggestions, classify it as "Suggestion".  
Otherwise, if the feedback content falls under the following categories, classify it as ICC.  
ICC stands for Issues (errors, failures, defects), Complaints (dissatisfaction, inconvenience, discomfort), and Comments (simple inquiries, opinions), representing general problem reports or feedback.

Classification criteria:
- If feedback is a new feature request or improvement suggestion: "Suggestion"
- If feedback contains or implies the following concepts: "ICC"
  - Errors
  - Failures
  - Defects
  - Complaints
  - Inconvenience
  - Discomfort
  - Simple inquiries
  - General opinions

Output format:
- If classified as Suggestion: "Suggestion"
- If classified as ICC: "ICC"

Do not include any additional explanations or supplementary text.

Input Summary:
{afterchange}
""")

### Step 2: Suggestion Separation
split_prompt = PromptTemplate.from_template("""<Role> You extract and organize multiple user suggestions from text. </Role>
<Task>: Your task is to separate each distinct suggestion clearly. Identify and number each distinct suggestion in the user's summarized text.

Reasoning: Use a Thought step to identify if the input contains more than one distinct idea (look for keywords like "and", punctuation, or multiple sentences indicating separate ideas). Then Act by listing each suggestion separately in the required format. Keep each suggestion phrased as a standalone improvement point.

Text:
{afterchange}

<Format>: If there are multiple suggestions in the input, output them as a JSON array of strings, where each string is one suggestion. (e.g., ["Suggestion 1", "Suggestion 2"]). Ensure the JSON is valid. If there is only one suggestion, still output a single-element JSON array with that suggestion. Do not add any explanatory text, just the JSON.""")

### Step 3: Overall Suggestion Summary
overall_summary_prompt = PromptTemplate.from_template("""
<Role> You are a home appliance expert, specialized at summarizing electronic product suggestions comprehensively. </Role>

<Task> Let's think step by step.

1. Understand all the suggestions in `{proposals}` thoroughly.
2. For each suggestion, summarize it comprehensively while maintaining its distinctiveness and context, especially with respect to the component `{components}`.
3. Do not merge different suggestions into one - keep them as separate suggestions if they are distinct.

Strictly follow the format below. </Task>

<Format> Please follow these strict output rules:

- Output must be a valid JSON array of strings.
- The output must start with `[` and end with `]`, and contain no other text before or after.
- Do not include any explanations, labels, metadata, or extra text.
- Each string in the array should be a distinct summarized suggestion.
</Format>
""")

### Step 4: First Suggestion Selection
first_prompt = PromptTemplate.from_template("""<Role> You select the first suggestion from the comprehensive summary based on order of appearance. </Role>
<Task>: Identify and return the first suggestion from the comprehensive summary, exactly as it appears, without any additional text.

Reasoning: Use a Thought step to parse the comprehensive summary and find the first suggestion mentioned. Then Act by outputting that first suggestion verbatim. Do not include any formatting, numbering, or other suggestions.

Comprehensive Summary: 
{overall_summary}  

List of Original Suggestions:
{proposals}

<Format>: Output the first Suggestion as a plain text string (no JSON, no list formatting, no quotes around it).""")

### Step 5: Similarity Calculation (Not calling LLM)

### Step 6: Self-Consistency Evaluation (Structured Assessment Criteria)
sc_prompt = PromptTemplate.from_template("""
<Role> You are an expert who independently evaluates the most similar document based on the Suggestion statement and top 10 similarity documents. This process will be repeated 3 times to ensure Self-Consistency. </Role>

<Task>: Based on the following information, select the most similar document. Evaluate each criterion below with numeric scores (integers), and calculate the total evaluation score and thinking confidence at the end.

[Input Information]
Suggestion:
{proposal_summary}

Top 10 Documents:
{top_10_table}

<Evaluation Criteria>: Evaluate each criterion with quantitative scores.

- Selected Document (Ticket_id): [e.g., fdff64d]
- Functional Category Alignment (Context Entity Recall): [0-25 points]
- Claim Coverage (Context Recall): [0-25 points]
- Evidence Faithfulness: [0-25 points]
- Explanation Flow Similarity: [0-25 points]

<Qualitative Assessment Criteria>: Self-evaluate your reasoning process.

- Clarity of Criteria Application: [0-30 points]
- Logical Coherence: [0-30 points]
- Persuasiveness of Explanation: [0-20 points]
- Absence of Ambiguity: [0-20 points]

- Total Evaluation Score: [Sum of above four items, 0-100 points]
- Thinking Confidence (Self-Eval): [Convert qualitative total to percentage]

<Output Format>: Follow the format below exactly.

[Response #1]
- Selected Document: fdff61d
- Functional Category Alignment (Context Entity Recall): 25 points
- Claim Coverage: 20 points
- Evidence Faithfulness: 24 points
- Explanation Flow Similarity: 22 points
- Total Evaluation Score: 91 points
- Clarity of Criteria Application: 30 points
- Logical Coherence: 28 points
- Persuasiveness of Explanation: 18 points
- Absence of Ambiguity: 20 points
- Thinking Confidence (Self-Eval): 96%
""")

### Step 7: Final Decision
choose_prompt = PromptTemplate.from_template("""<Role7> You are an expert who integrates three repeated evaluation results and determines the most reliable final recommended document. </Role7>

<Task>: Analyze the three Self-Consistency-based evaluation responses together with RAG-based similarity scores and RAGAS-based quantitative metrics to ultimately select the most similar document and explain your reasoning.

Reasoning: Collect all the following metrics to calculate an integrated score, then select the most comprehensively similar document.

<Integrated Evaluation Criteria>
1. Self-Consistency Average Confidence (Weight 0.3)
2. Same Document Repeated Selection (+10 point bonus)
3. Self-Consistency Average Total Score (Weight 0.2)
4. RAG-based Cosine Similarity (Weight 0.2)
5. Faithfulness (RAGAS metric) (Weight 0.15)
6. Context Recall (RAGAS metric) (Weight 0.15)

<Final Selection Criteria>
- Select the document with the highest integrated score.
- In case of equal scores, prioritize documents more frequently selected in Self-Consistency.
- The decision should be based on how similar the document summary, functional purpose, and explanation style are to the Suggestion statement.

Self-Consistency Response List:
{self_consistency_responses}

RAG Similarity List:
{top_10_table}

<Format>: Output following the format below.

> ✅ Final Recommended Document: [Document Ticket ID (e.g., d25d98b)]
> 📄 Selected Document Summary: [Core content summary of the selected document]
> 🔒 Recommendation Confidence (%): [0-100 numeric value, based on repeated responses and average confidence]
> 📊 RAG-based Similarity (%): [RAG similarity of selected document, or "N/A" if not available]
> 🧠 Selection Rationale Summary:
> - Same document repeatedly selected in Self-Consistency responses (+10 points)
> - Highest average confidence and evaluation scores
> - Superior RAG similarity and Faithfulness/Context Recall scores
> - Most similar functional purpose and narrative style to the Suggestion statement
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
    index = original_vectordb.index
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
    "afterchange": inputs["afterchange"]
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
    RunnableLambda(timed("입력받기")(lambda x: x)),
    RunnableLambda(timed("ICC 분류")(lambda x: x | {
        "icc_check": (icc_prompt | llm).invoke({"afterchange": x["afterchange"]}).content
    })),
    RunnableLambda(timed("ICC 분기처리")(lambda x: (
        (update_icc_df(x) or print("👉 판별 결과 : ICC\n <종료>") or exit(0))
    ) if x["icc_check"] == "ICC" else (
        (print("👉 판별 결과 : Proposal") or x)
    ))),
    RunnableLambda(timed("제안 분리")(lambda x: x | {
        "proposals": (split_prompt | llm).invoke({"afterchange": x["afterchange"]}).content
    })),
    RunnableLambda(timed("전체 제안 요약")(lambda x: x | {
        "proposal_summary_all": (
            lambda response: (
                json.loads(
                    re.search(r"\[.*\]", response, re.DOTALL).group(0)
                ) if re.search(r"\[.*\]", response, re.DOTALL)
                else [response.strip()]
            ))(
                (lambda resp: (
                    resp.content if hasattr(resp, "content") else str(resp)
                ))(
                    llm.invoke(overall_summary_prompt.format(
                        proposals=x["proposals"],
                        components=x["components"]
                    ))
                )
            )
    })),
    RunnableLambda(timed("첫번째 제안 선택")(lambda x: x | {
        "first_proposal": (first_prompt | llm).invoke({
            "overall_summary": x["proposal_summary_all"],
            "proposals": json.dumps([x["proposal_summary_all"]])
        }).content
    })),
    RunnableLambda(timed("Top-10 유사 문서 검색")(lambda x: x | {
        "top_10_table": retrieve_context(x["first_proposal"])[0],
        "top_10_docs": retrieve_context(x["first_proposal"])[1],
        "top_10_records": retrieve_context(x["first_proposal"])[2]
    })),
    RunnableLambda(timed("Self-Consistency 평가")(lambda x: x | {
        "self_consistency_responses": [
            llm.invoke(sc_prompt.format(
                proposal_summary=x["first_proposal"],
                top_10_table=format_top10_for_prompt(x["top_10_docs"])
            )) for _ in range(3)
        ]
    })),
    RunnableLambda(timed("Self-Consistency 파싱")(lambda x: x | {
        "parsed_self_consistency": [
            parse_self_consistency_response(resp.content if hasattr(resp, "content") else str(resp))
            for resp in x["self_consistency_responses"]
        ]
    })),
    RunnableLambda(timed("Self-Consistency 테이블 포맷")(lambda x: x | {
        "self_consistency_table": format_self_consistency_table(x["parsed_self_consistency"])
    })),
    RunnableLambda(timed("최종 결정")(lambda x: x | {
        "final_result": (choose_prompt | llm).invoke({
            "self_consistency_responses": json.dumps(
                [resp.content if hasattr(resp, "content") else str(resp) for resp in x["self_consistency_responses"]]
            ),
            "top_10_table": x["top_10_table"]
        }).content
    }))
)




# 모듈화: main() 함수 정의 및 반환값 구성
def main(inputs):
    global timing_results
    timing_results = []
    print("\n📥 입력 제안문:")
    print(inputs["afterchange"])

    result = chain.invoke(inputs)
    print("✅ result keys:", result.keys())

    print("\n📄 전체 제안 요약:")
    print(result["proposal_summary_all"])

    print("\n⭐ 첫 번째 제안:")
    print(result["first_proposal"])

    print("\n🔝 Top-10 유사 문서 테이블:")
    print(result["top_10_table"])

    print(result["final_result"])
    
    # RAGAS 점수 추출
    import re
    final_text = result["final_result"]
    match_ragas = re.search(r"RAG-based Similarity.*?:\s*([0-9.]+)", final_text)
    ragas_score = float(match_ragas.group(1)) if match_ragas else 0

    # 선택된 문서의 ticket_id 추출
    match_ticket = re.search(r"Final Recommended Document:\s*(\w+)", final_text)
    ticket_id = match_ticket.group(1) if match_ticket else ""

    selected_row = df[df["ticket_id_hashed"] == ticket_id]
    if not selected_row.empty:
        keyword = selected_row.iloc[0].get("keyword", "N/A")
        before_change = selected_row.iloc[0].get("before_change", "")
        #after_change = selected_row.iloc[0].get("after_change", "")
        after_change = inputs.get("afterchange", "")
    else:
        keyword = "N/A"
        before_change = ""
        after_change = ""

    components = result.get("components", inputs.get("components", ""))
    generated_summary = result.get("first_proposal", inputs.get("generated_summary", ""))
    proposal_summary_all = result.get("proposal_summary_all", inputs.get("proposal_summary_all", []))
    
    return ticket_id, components, before_change, after_change, generated_summary, ragas_score, keyword, timing_results, proposal_summary_all

if __name__ == "__main__":
    main()
