import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain.chains import ConversationalRetrievalChain
from langchain.evaluation.criteria import LabeledCriteriaEvalChain

# 1. Streamlit UI
st.set_page_config(page_title="EchoDeepak: Socratic Mentor", layout="wide")
st.title("🧠 EchoDeepak: Socratic GenAI Mentor")
st.markdown("Choose a topic to have a Socratic dialogue and get your score at the end!")

topics = [
    "Prompt Engineering",
    "Few-shot / One-shot / CoT",
    "LangChain / LlamaIndex",
    "Retrieval-Augmented Generation (RAG)",
    "Hallucinations in LLMs",
    "Responsible AI",
    "Agents and Automation",
    "GenAI Use Cases",
    "Ethics and Risks"
]

selected_topic = st.selectbox("🎯 Select a GenAI Topic", topics)
start_convo = st.button("Start Socratic Conversation")

# 2. Define System Prompt
system_prompt = f'''
You are EchoDeepak, a Socratic AI mentor on the HiDevs platform, and you respond in english only.
Your goal is to help students think deeply and clearly about Generative AI topics.

Do not give direct answers. Instead, respond with thoughtful follow-up questions that:
- Challenge assumptions
- Ask for real-world examples
- Encourage clarity, logic, and reasoning
- Explore consequences, comparisons, and counterpoints

Avoid being too formal. Keep it conversational, engaging, and encouraging.

The student has chosen the topic: {selected_topic}.

Each response should end with 1 or 2 new questions that help the student go deeper.

If the student gives vague or surface-level answers, gently ask for clarification, examples, or alternative views.

NEVER reveal the answer. Your job is to guide them to discover the truth themselves.
'''

# 3. Load Gemini & Qdrant (Mock setup for demo)
@st.cache_resource
def setup_chain():
    # Initialize LLM and Embeddings
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.4, system_message=system_prompt)
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Semantic Chunking
    text = "Mock GenAI knowledge base content for testing RAG pipeline on chosen topic."
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)

    # Qdrant setup
    qdrant = QdrantClient(":memory:")
    qdrant.recreate_collection(
        collection_name="genai_demo",
        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
    )

    doc_store = Qdrant(
        client=qdrant,
        collection_name="genai_demo",
        embeddings=embedding_model
    )
    doc_store.add_texts(chunks)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=doc_store.as_retriever(search_type="mmr", k=3),
        return_source_documents=False
    )
    return qa_chain, llm

# 4. Initialize QA chain
qa_chain, llm = setup_chain()

# 5. Session State Setup
if "history" not in st.session_state:
    st.session_state.history = []

if "conversation" not in st.session_state:
    st.session_state.conversation = ""

if "conversation_started" not in st.session_state:
    st.session_state.conversation_started = False

# 6. Start Conversation Logic
if start_convo:
    st.session_state.conversation_started = True

if st.session_state.conversation_started:
    user_input = st.text_input("👤 You:", key="user_input", placeholder="Ask something related to the topic...")

    if user_input:
        response = qa_chain({"question": user_input, "chat_history": st.session_state.history})
        st.session_state.history.append((user_input, response["answer"]))
        st.session_state.conversation += f"User: {user_input}\nEchoDeepak: {response['answer']}\n"
        st.markdown(f"**EchoDeepak:** {response['answer']}")

    if st.session_state.history:
        with st.expander("🗒 Full Conversation"):
            for q, a in st.session_state.history:
                st.markdown(f"**You:** {q}")
                st.markdown(f"**EchoDeepak:** {a}")

# 7. Evaluation Button
if st.button("🎯 Evaluate My Understanding"):
    evaluator = LabeledCriteriaEvalChain.from_llm(
        llm=llm,
        criteria={
            "clarity": "Was the student's explanation easy to follow and coherent?",
            "depth": "Did they go beyond surface-level definitions to explore reasoning, implications, or challenges?",
            "application": "Did they connect the concept to a real-world use case or startup scenario?",
            "critical_thinking": "Did they reflect on assumptions, limitations, or offer counterpoints?",
            "progression": "Did their understanding deepen as the conversation evolved?",
            "relevance": "Did they remain on-topic and avoid fluff?",
            "creativity": "Did they provide a fresh, original, or unique view?",
        }
    )

    score = evaluator.evaluate_strings(
        input="The user engaged in a conversation about Generative AI.",
        prediction=st.session_state.conversation,
        reference="A deep, well-reasoned conversation exploring the GenAI concept in a structured, logical, and insightful manner."
    )

    st.subheader("📊 Evaluation Report")
    st.write(score["score"])


