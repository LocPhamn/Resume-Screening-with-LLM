import glob
import os
import re
from langchain_core.documents import Document
import numpy as np
from google import genai
from langchain_huggingface import HuggingFaceEmbeddings
from pypdf import PdfReader
from dotenv import load_dotenv
from tqdm import tqdm
import json
from langchain_community.vectorstores import FAISS
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore


vecto_db_path = r"D:\Python plus\NLP_AI\src\nlp\proj\resume_evaluation-upgrade\faiss_index"
load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
embedder = HuggingFaceEmbeddings(model_name="bkai-foundation-models/vietnamese-bi-encoder", )
faiss_db = FAISS.load_local(vecto_db_path,embedder, allow_dangerous_deserialization=True)


def extract_text_from_pdf(file_path):
    """
    Extract text from a PDF file
    :return: documents as a list of strings
    """
    docs = ""
    reader = PdfReader(file_path)
    for page in reader.pages:
        text = page.extract_text()
        if text:
            docs += text
        else:
            print("No text found on this page.")
    return docs.strip()


def create_embeddings(text, embedder=None):
    if embedder is None:
        embedder = HuggingFaceEmbeddings(model_name="bkai-foundation-models/vietnamese-bi-encoder", )
    return embedder.embed_query(text)


def l2_distance(vec1, vec2):
    # Compute the dot product of the two vectors and divide by the product of their norms
    return np.linalg.norm(vec1 - vec2)


def chunk_text(texts, n, overlap):
    chunk = []
    for i in range(0, len(texts), n - overlap):
        chunk.append(texts[i:i + n])
    return chunk


def store_to_faiss(jd_name,chunks, embeddings_chunk, path=vecto_db_path):
    for i, (chunk, embedding) in tqdm(enumerate(zip(chunks, embeddings_chunk))):
        faiss_db.add_texts(
            texts=[chunk],
            embeddings=[embedding],
            metadatas=[{"index": i, "name": jd_name}],
        )
    faiss_db.save_local(path)
    return faiss_db


def response_to_json(response, path=None):
    """
    Convert the response text to a structured JSON format.
    :param response: The response text from the model.
    :return: A dictionary containing the structured data.
    """
    metadata = {}
    pattern = r"\*\*\s*(\w+):\*\*\s*(.+)"
    if not response or path is None:
        return

    for line in response.strip().split("\n"):
        match = re.search(pattern, line.strip())
        if match:
            key = match.group(1).strip()
            value = match.group(2).strip()
            # If the value contains commas, treat it as a list
            if ',' in value and key == "skill":
                value = [v.strip() for v in value.split(',')]
            metadata[key] = value
    with open(os.path.join(r"D:\Python plus\NLP_AI\src\rag\jd", path), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)
    return metadata


def resume_to_json(resume_text, vecto_db):
    jsonstr = json.loads(resume_text.split('```json')[1].split('```')[0].strip())
    text = "{}".format(jsonstr)
    if jsonstr is not None:
        document = Document(
            page_content=text,
            metadata={"cv_id": "resume_data"}
        )
        vecto_db.add_documents(
            documents=[document],
        )
        vecto_db.save_local(vecto_db_path)


# Rerank Retrieved Results
def rerank_results(query, results, top_n=3, model="gemini-2.0-flash", path=None):
    print(f"Reranking {len(results)} documents...")  # Print the number of documents to be reranked

    scored_results = []  # Initialize an empty list to store scored results

    # Define the system prompt for the LLM
    system_prompt = """You are an expert at evaluating document relevance for search queries.
    Your task is to rate documents on a scale from 0 to 10 based on how well they answer the given query.

    Guidelines:
    - Score 0-2: Document is completely irrelevant
    - Score 3-5: Document has some relevant information but doesn't directly answer the query
    - Score 6-8: Document is relevant and partially answers the query
    - Score 9-10: Document is highly relevant and directly answers the query

    You MUST respond with ONLY a single integer score between 0 and 10. Do not include ANY other text."""

    for i, (result, similarity_score) in enumerate(results):
        # Show progress every 5 documents
        if i % 5 == 0:
            print(f"Scoring document {i + 1}/{len(results)}...")

        # Define the user prompt for the LLM
        user_prompt = f"""Query: {query}

        Document:
        {result.page_content}

        Rate this document's relevance to the query on a scale from 0 to 10:"""

        # Get the LLM response
        response = client.models.generate_content(
            model=model,
            contents=system_prompt + user_prompt
        )

        # Extract the score from the LLM response
        score_text = response.text.strip()  # Clean up the response text
        print(f"LLM Response: {score_text}")  # Print the LLM response for debugging
        # Use regex to extract the numerical score
        score_match = re.search(r'\b(10|[0-9])\b', score_text)
        if score_match:
            score = float(score_match.group(1))
        else:
            # If score extraction fails, use similarity score as fallback
            print(f"Warning: Could not extract score from response: '{score_text}', using similarity score instead")
            score = result["similarity"] * 10

        # Append the scored result to the list
        scored_results.append({
            "text": result.page_content,
            "metadata": result.metadata,
            "similarity": similarity_score,
            "relevance_score": score
        })

        # Sort results by relevance score in descending order
    reranked_results = sorted(scored_results, key=lambda x: x["relevance_score"], reverse=True)

    # Return the top_n results
    return reranked_results[:top_n]


def jd_requirement(jd_path,reranking_method="llm", top_n=3, model="gemini-2.0-flash",
                   path=None):
    jd_name = os.path.basename(str(jd_path)).split(".")[0]
    jd_text = extract_text_from_pdf(jd_path)
    chunks = chunk_text(jd_text, 200, 30)
    chunk_embeddings = [create_embeddings(chunk, embedder) for chunk in tqdm(chunks)]
    store_to_faiss(jd_name,chunks, chunk_embeddings, path=vecto_db_path)

    jd_query = f"""
    Tôi muốn bạn tạo ra metadata gồm các trường như: 'skill', 'education', 'certification', 'soft_skill', 'experience' từ job description"""
    jd_system_promt = "bạn là một trợ lý AI thông minh có khả năng tạo metadata cho job description hãy sử dụng thông tin job description trong context tạo ra metadata theo query nếu trường thông tin nào không có hãy trả lời là không yêu cầu . Trả lời bằng tiếng Việt."

    initial_results = faiss_db.similarity_search_with_score(jd_query, k=10, filter={"name": jd_name})
    if reranking_method == "llm":
        reranked_results = rerank_results(jd_query, initial_results, top_n=top_n)
    else:
        reranked_results = initial_results[:top_n]

    context = "\n\n===\n\n".join([result["text"] for result in reranked_results])

    # Generate response based on context
    user_prompt = f"""
                Context:
                {context}

                Question: {jd_query}

            """

    response = generate_response(jd_system_promt, user_prompt, model)
    response_to_json(response, path=path)

    return {
        "query": jd_query,
        "reranking_method": reranking_method,
        "initial_results": initial_results[:top_n],
        "reranked_results": reranked_results,
        "context": context,
        "response": response
    }


def extract_candidate_info(text, model="gemini-2.0-flash"):
    eval_system_promt = f"""
            Là một Hệ thống AI Đánh giá resume. có kỹ năng với kiến thức chuyên sâu về công nghệ và công nghệ thông tin, vai trò của bạn là đánh giá tỉ mỉ sơ yếu lý lịch của ứng viên dựa trên mô tả công việc được cung cấp.

            Đánh giá của bạn sẽ bao gồm việc phân tích sơ yếu lý lịch để tìm các kỹ năng, kinh nghiệm và trình độ phù hợp với yêu cầu công việc. Tìm kiếm các từ khóa chính và tiêu chí cụ thể được nêu trong mô tả công việc để xác định ứng viên có phù hợp với vị trí này hay không.

            Cung cấp đánh giá chi tiết về mức độ phù hợp của sơ yếu lý lịch với các yêu cầu công việc, nêu bật điểm mạnh, điểm yếu và bất kỳ lĩnh vực nào có thể quan tâm.

            Đánh giá của bạn phải toàn diện, chính xác và khách quan, đảm bảo rằng các ứng viên đủ điều kiện nhất được xác định chính xác dựa trên nội dung sơ yếu lý lịch của họ liên quan đến tiêu chí công việc.

            Hãy nhớ sử dụng chuyên môn của bạn về công nghệ và công nghệ thông tin để tiến hành đánh giá toàn diện nhằm tối ưu hóa quy trình tuyển dụng cho công ty tuyển dụng. Những hiểu biết sâu sắc của bạn sẽ đóng vai trò quan trọng trong việc xác định ứng viên có phù hợp với vai trò công việc hay không.
           """
    # Generate response based on context
    user_prompt = f"""
        Từ sơ yếu lý lịch sau: {text}
        Hãy muốn bạn tạo ra metadata gồm các trường như:'name','number','email', 'skill', 'education', 'certification', 'soft_skill', 'experience'
        Đầu ra metadata:
        1. name: [Tên ứng viên]
        2. number: [Số điện thoại ứng viên]
        3. email: [Email ứng viên]
        4  position: [Vị trí ứng tuyển của ứng viên]
        5. skill: [Các kỹ năng của ứng viên]
        6. education: [Trình độ học vấn của ứng viên]
        7. certification: [Các chứng chỉ của ứng viên]
        8. soft_skill: [Các kỹ năng mềm của ứng viên]
        9. experience: [Kinh nghiệm làm việc của ứng viên]. 
        output phải đúng định dạng này đừng cố tạo định dạng khác. đây là ví dụ về định dạng đầu ra:

        "name": "PHAM NGOC LOC",
        "number": "098-145-8737",
        "email": "jasonpham20203@gmail.com",
        "position": "AI INTERN",
        "skill": [
            "Computer Vision",
            "Natural Language Processing",
            "Machine Learning",
            "Deep Learning",
            "Python",
            "JavaScript",
            "Java",
            "PyTorch",
            "TensorFlow",
            "OpenCV",
            "spaCy",
            "Scikit-learn",
            "Streamlit",
            "MySQL",
            "SQL",
            "GIT",
            "Docker"
        ],
        "education": [
            "Major in Information Technology, Hanoi University of Architecture (2021 - now), GPA: 3.05/4.00"
        ],
        "certification": [
            "TOEIC 795"
        ],
        "soft_skill": [
            "Microsoft Office"
        ],
        "experience": [
            "Face-Recognition-For-Attendance-System (Personal Project): Collect, preprocess dataset for VGG Face model; fine-tune VGG Face model for Vietnamese faces; implement VGG Face with MTCNN and FAS-Net.",
            "Resume Evaluation (Personal Project): Read and clean resume data from PDF; implement regex to extract information; develop scoring formula using sentence embeddings; design Streamlit interface."
        ]
"""

    response = generate_response(eval_system_promt, user_prompt, model)
    brief_info = json.loads(response.split('```json')[1].split('```')[0].strip())
    resume_to_json(response, faiss_db)
    return {
        "user_prompt": user_prompt,
        "eval_system_promt": eval_system_promt,
        "response": response,
        "brief_info": brief_info
    }


def evaluate_resume(jd):
    faiss_dbe = FAISS.load_local(vecto_db_path, embedder, allow_dangerous_deserialization=True)
    query = f"cho tôi ứng viên hàng đầu phù hợp với jd này {jd}"
    eval_system_promt = f"""
               Là một Hệ thống AI Đánh giá resume. có kỹ năng với kiến thức chuyên sâu về công nghệ và công nghệ thông tin, vai trò của bạn là đánh giá tỉ mỉ sơ yếu lý lịch của ứng viên dựa trên mô tả công việc được cung cấp.

               Đánh giá của bạn sẽ bao gồm việc phân tích sơ yếu lý lịch để tìm các kỹ năng, kinh nghiệm và trình độ phù hợp với yêu cầu công việc. Tìm kiếm các từ khóa chính và tiêu chí cụ thể được nêu trong mô tả công việc để xác định ứng viên có phù hợp với vị trí này hay không.

               Cung cấp đánh giá chi tiết về mức độ phù hợp của sơ yếu lý lịch với các yêu cầu công việc, nêu bật điểm mạnh, điểm yếu và bất kỳ lĩnh vực nào có thể quan tâm.

               Đánh giá của bạn phải toàn diện, chính xác và khách quan, đảm bảo rằng các ứng viên đủ điều kiện nhất được xác định chính xác dựa trên nội dung sơ yếu lý lịch của họ liên quan đến tiêu chí công việc.

               Hãy nhớ sử dụng chuyên môn của bạn về công nghệ và công nghệ thông tin để tiến hành đánh giá toàn diện nhằm tối ưu hóa quy trình tuyển dụng cho công ty tuyển dụng. Những hiểu biết sâu sắc của bạn sẽ đóng vai trò quan trọng trong việc xác định ứng viên có phù hợp với vai trò công việc hay không.
              """
    print(f"Running similarity search for query: {query}")
    results = faiss_dbe.similarity_search_with_score(query, k=12, filter={"cv_id": "resume_data"})
    for i, (result, similarity_score) in enumerate(results):
        print(result.page_content)

    print(f"Found {len(results)} results for query: {query}")

    reranked_results = rerank_results(query, results, top_n=4)
    context = "\n\n===\n\n".join([result['text'] for result in reranked_results])
    user_prompt = f"""
            hãy dựa vào thông tin các ứng viên cung cấp từ context và đánh giá ứng viên nào phù hợp với yêu cầu công việc nhất
            context: {context}
            job requirement: {jd}.

            Đầu ra đánh giá:
            1. Tính tỷ lệ phần trăm khớp giữa sơ yếu lý lịch và mô tả công việc. Đưa ra một con số và một số giải thích
            2. dựa vào tỷ lệ phần trăm khớp giữa sơ yếu lý lịch và mô tả công việc, hãy kết luận resume này phù hợp hay không phù hợp với yêu cầu công việc. trả lời  1 nếu phù hợp và 0 nếu không phù hợp vừa đưa ra lý do giải thích quyết định.
            nếu phù hợp hãy tạo metadata  gồm các trường như sau: 'name: [ Tên Ưng viên]', 'suitable_rate: [tỉ lệ khớp]','candidate_strength':[các điểm mạnh của ứng viên] ,'reason: [những lí do phù hợp]'
            """
    response = generate_response(eval_system_promt, user_prompt, model="gemini-2.0-flash")
    return {
        "user_prompt": user_prompt,
        "eval_system_promt": eval_system_promt,
        "response": response
    }


def generate_response(system_promt, user_prompt, model="gemini-2.0-flash"):
    """
    Generate a response using the specified model.

    Args:    query (str): The user's query.
    context (str): The context or retrieved documents to provide additional information.
    model (str): The model to use for generating the response.
    Returns:
    str: The generated response from the model.
    """

    response = client.models.generate_content(
        model=model,
        contents=system_promt + user_prompt
    )
    return response.text

def brief_resume(pdf_path, model="gemini-2.0-flash"):

    if not pdf_path:
        return {}

    resume_text = extract_text_from_pdf(pdf_path)
    response = extract_candidate_info(resume_text,model=model)

    return response

def run():
    # file_paths = [
    #     r"D:\Python plus\NLP_AI\src\rag\data\TTS_Java.pdf",
    # ]
    vector_db_path = r"D:\Python plus\NLP_AI\src\nlp\proj\resume_evaluation-upgrade\faiss_index"

    # json_path = r"D:\Python plus\NLP_AI\src\rag\jd\{}.json".format(os.path.basename(file_paths[0]).split(".")[0])
    # job_description = extract_text_from_pdf(file_paths[0])
    #
    # resume_path = r"D:\Python plus\NLP_AI\src\nlp\proj\resume_evaluation-upgrade\resume\*.pdf"
    # resumes = glob.glob(resume_path)
    #
    # chunks = chunk_text(job_description, 200, 30)
    # chunk_embeddings = [create_embeddings(chunks, embedder) for chunks in tqdm(chunks)]


    # faiss_db = store_to_faiss(faiss_db, chunks, chunk_embeddings, path=vector_db_path)

    # jd_requirement(jd_query, jd_system_promt, faiss_db, reranking_method="llm", top_n=3, model="gemini-2.0-flash",
    #                path=json_path)

    
    #
    faiss_db = FAISS.load_local(vector_db_path, embedder, allow_dangerous_deserialization=True)
    with open(r"/jd/AI Engineer Fresher.json", "r", encoding="utf-8") as f:
        jd = json.load(f)

    query = f"cho tôi ứng viên hàng đầu phù hợp với jd này {jd}"
    eval_system_promt = f"""
               Là một Hệ thống AI Đánh giá resume. có kỹ năng với kiến thức chuyên sâu về công nghệ và công nghệ thông tin, vai trò của bạn là đánh giá tỉ mỉ sơ yếu lý lịch của ứng viên dựa trên mô tả công việc được cung cấp.

               Đánh giá của bạn sẽ bao gồm việc phân tích sơ yếu lý lịch để tìm các kỹ năng, kinh nghiệm và trình độ phù hợp với yêu cầu công việc. Tìm kiếm các từ khóa chính và tiêu chí cụ thể được nêu trong mô tả công việc để xác định ứng viên có phù hợp với vị trí này hay không.

               Cung cấp đánh giá chi tiết về mức độ phù hợp của sơ yếu lý lịch với các yêu cầu công việc, nêu bật điểm mạnh, điểm yếu và bất kỳ lĩnh vực nào có thể quan tâm.

               Đánh giá của bạn phải toàn diện, chính xác và khách quan, đảm bảo rằng các ứng viên đủ điều kiện nhất được xác định chính xác dựa trên nội dung sơ yếu lý lịch của họ liên quan đến tiêu chí công việc.

               Hãy nhớ sử dụng chuyên môn của bạn về công nghệ và công nghệ thông tin để tiến hành đánh giá toàn diện nhằm tối ưu hóa quy trình tuyển dụng cho công ty tuyển dụng. Những hiểu biết sâu sắc của bạn sẽ đóng vai trò quan trọng trong việc xác định ứng viên có phù hợp với vai trò công việc hay không.
              """
    print(f"Running similarity search for query: {query}")
    results = faiss_db.similarity_search_with_score(query, k=12, filter={"cv_id": "resume_data"})
    for i, (result, similarity_score) in enumerate(results):
        print(result.page_content)

    print(f"Found {len(results)} results for query: {query}")

    reranked_results = rerank_results(query, results, top_n=4)
    context = "\n\n===\n\n".join([result['text'] for result in reranked_results])
    user_prompt = f"""
            hãy dựa vào thông tin các ứng viên cung cấp từ context và đánh giá ứng viên nào phù hợp với yêu cầu công việc nhất
            context: {context}
            job requirement: {jd}.

            Đầu ra đánh giá:
            1. Tính tỷ lệ phần trăm khớp giữa sơ yếu lý lịch và mô tả công việc. Đưa ra một con số và một số giải thích
            2. dựa vào tỷ lệ phần trăm khớp giữa sơ yếu lý lịch và mô tả công việc, hãy kết luận resume này phù hợp hay không phù hợp với yêu cầu công việc. trả lời  1 nếu phù hợp và 0 nếu không phù hợp vừa đưa ra lý do giải thích quyết định.
            nếu phù hợp hãy tạo metadata  gồm các trường như sau: 'name: [ Tên Ưng viên]', 'suitable_rate: [tỉ lệ khớp]','candidate_strength':[các điểm mạnh của ứng viên] ,'reason: [những lí do phù hợp]'
            """
    response = generate_response(eval_system_promt, user_prompt, model="gemini-2.0-flash")
    print("Response:", response)
if __name__ == '__main__':
    run()





