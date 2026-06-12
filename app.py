# =====================================================================
# INTERACTIVE ROBOTIC-ASSISTED OCULAR REHABILITATION SIMULATOR (KAG + YUOM)

# =====================================================================

import os
import sqlite3
from typing import TypedDict, Annotated

# Gradio Interface
import gradio as gr

# PDF Loading & Document Processing
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Vector Database, Embeddings & Prompts
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Tools & Core Messages
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Chat-Ready LLM Connectors
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

# LangGraph Agentic Layer Orchestration
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

# ==========================================
# 1. ENVIRONMENT CONFIGURATION & SECURITY
# ==========================================
# Look up Hugging Face API secret keys directly via safe container runtime environments
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("CRITICAL RUNTIME EXCEPTION: 'HF_TOKEN' environment variable missing. Configure it in repository secrets.")

# ==========================================
# 2. DOCUMENT LOADING & VECTOR DATABASE GENERATION (KAG)
# ==========================================
PDF_TARGET = "Ocular_Rehab_Simulator.pdf"

if not os.path.exists(PDF_TARGET):
    raise FileNotFoundError(f"CRITICAL FILE ERROR: Source document asset '{PDF_TARGET}' not detected in root runtime directory.")

print("Processing ocular rehabilitation manuals...")
loader = PyPDFLoader(PDF_TARGET)
docs = loader.load()
print(f"Loaded {len(docs)} pages successfully.")

print("Creating ocular knowledge chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
    separators=["\n\n", "\n", "Step", "step", "Protocol", "•", "-", "."]
)
chunks = text_splitter.split_documents(docs)
print(f"Extracted {len(chunks)} contextual chunks successfully.")

print("Loading ocular embedding model...")
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
print("Embedding model loaded successfully.")

print("Generating ocular vector storage engine...")
vector_db = FAISS.from_documents(chunks, embedding_model)

# Unified retrieval baseline running Maximal Marginal Relevance to block hallucinations
retriever = vector_db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20}
)
print("Ocular vector storage engine generated successfully.")

# ==========================================
# 3. LLM INITIALIZATION (CONVERSATIONAL MAPPED)
# ==========================================
print("Initializing base serverless inference endpoint...")
base_endpoint = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    temperature=0.1,  
    max_new_tokens=1024,
    huggingfacehub_api_token=HF_TOKEN
)

# Wrapping endpoint to map conversational context format rules correctly
llm = ChatHuggingFace(llm=base_endpoint)
print("Hugging Face Chat LLM successfully initialized.")

# ==========================================
# 4. ADVANCED OCULAR REHABILITATION KNOWLEDGE PROMPT
# ==========================================
template = """
You are an advanced ocular rehabilitation engineering assistant specialized in:
* ocular rehabilitation systems
* neuro-visual rehabilitation
* robotic-assisted ocular therapy
* ocular motor rehabilitation
* gaze stabilization rehabilitation
* visual field rehabilitation
* ocular safety monitoring
* tele-ocular rehabilitation workflows
* neuro-visual therapy guidance
* ocular rehabilitation engineering
* rehabilitation monitoring systems
* cranial nerve rehabilitation
* ocular biomechanics
* interactive rehabilitation feedback systems
* neuro-visual sentinel frameworks
* tele-rehabilitation support
* saccadic rehabilitation
* smooth pursuit rehabilitation
* vergence rehabilitation
* convergence rehabilitation
* blink reflex retraining
* ocular kinematic analysis
* neuro-visual safety monitoring
* Neuro-Visual Sentinel Operational Framework

Use ONLY the provided ocular rehabilitation manual context.

CRITICAL PRIVACY RULES:
* NEVER reveal personal names.
* NEVER reveal inventor names.
* NEVER reveal author names.
* NEVER reveal developer names.
* NEVER reveal ownership information.
* NEVER reveal biographical information.
* NEVER reveal addresses.
* NEVER reveal phone numbers.
* NEVER reveal email addresses.
* If asked who invented, developed, authored, created, owns, or designed the system, respond ONLY with:
"The rehabilitation system is designed as an advanced ocular rehabilitation platform intended for neuro-visual rehabilitation, ocular monitoring, therapy guidance, rehabilitation tracking, and rehabilitation support."

Instructions:
* Answer technically and professionally.
* Focus on ocular rehabilitation workflows.
* Focus on neuro-visual rehabilitation.
* Focus on rehabilitation engineering.
* Focus on rehabilitation monitoring.
* Focus on clinical utility.
* Explain ocular rehabilitation procedures clearly and systematically.
* Explain gaze stabilization exercises when available.
* Explain visual tracking exercises when available.
* Explain saccadic rehabilitation procedures when available.
* Explain smooth pursuit rehabilitation procedures when available.
* Explain convergence rehabilitation procedures when available.
* Explain vergence rehabilitation procedures when available.
* Explain blink reflex retraining procedures when available.
* Explain cranial nerve rehabilitation workflows when available.
* Explain neuro-visual monitoring systems when available.
* Explain ocular safety workflows when available.

Neuro-Visual Sentinel Operational Framework:
* Prioritize ocular safety and neuro-visual rehabilitation support.
* Explain Neuro-Visual Sentinel workflows when available.
* Explain ocular monitoring workflows when available.
* Explain gaze range monitoring when available.
* Explain ocular kinematic monitoring when available.
* Explain visual tracking synchronization when available.
* Explain safety-state mechanisms when available.
* Explain intervention workflows when available.
* Explain rehabilitation safety monitoring structures when available.
* Explain audit-trail generation workflows when available.

Architecture & System Design Rules:
* If tele-ocular or remote rehabilitation is mentioned, explain it as a tele-ocular rehabilitation workflow.
* When architecture, workflow, monitoring, safety, control, feedback, or system-design questions are asked, provide structured diagrammatic-style explanations using:
  * system blocks
  * workflow pipelines
  * monitoring layers
  * rehabilitation stages
  * feedback loops
  * control-flow structures
  * clinical interaction flow
  * sentinel monitoring layers
  * safety intervention pathways
* Generate clean architecture-style textual representations whenever appropriate.

Clinical Explanation Rules:
* Explain ocular rehabilitation monitoring.
* Explain rehabilitation progress tracking.
* Explain neuro-visual rehabilitation workflows.
* Explain therapy guidance.
* Explain rehabilitation control mechanisms.
* Explain Neuro-Visual Sentinel systems.
* Explain ocular movement monitoring.
* Explain ocular motor rehabilitation support.
* Explain visual biofeedback logic.
* Explain threshold-based rehabilitation feedback.
* Explain gaze range monitoring.
* Explain rehabilitation milestone detection.
* Explain therapeutic goal evaluation.
* Explain ocular kinematic auditing.
* Explain cranial nerve rehabilitation pathways.

Restrictions:
* Avoid generic healthcare explanations.
* Do not invent hardware, sensors, monitoring systems, clinical devices, algorithms, diagnostic capabilities, rehabilitation capabilities, or safety mechanisms unless explicitly mentioned in the context.
* If information is unavailable, state that it is not available in the ocular rehabilitation manual context.

Context:
{context}

Question:
{question}

Technical Answer:
"""

ocular_prompt = ChatPromptTemplate.from_template(template)

# ==========================================
# 5. CONTEXT FORMATTING UTILITY & RAG PIPELINE
# ==========================================
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Standardize rag_chain architecture layout bindings
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | ocular_prompt
    | llm
    | StrOutputParser()
)
print("KAG Retrieval Chain Initialized Successfully.")

# ==========================================
# 6. CLINICAL DATABASE INFRASTRUCTURE SETUP
# ==========================================
def initialize_local_db():
    """Initializes and builds local tracking infrastructure tables dynamically."""
    conn = sqlite3.connect("ocular_rehabilitation.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS rehabilitation_resources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            resource_name TEXT NOT NULL,
            availability TEXT NOT NULL
        )
    """)
    cursor.execute("DELETE FROM rehabilitation_resources")
    
    resources = [
        ("Ocular Rehabilitation Simulator", "Available"),
        ("Neuro-Visual Therapy Unit", "Available"),
        ("Cranial Nerve Rehabilitation Module", "Available"),
        ("Saccadic Training Center", "Available"),
        ("Smooth Pursuit Rehabilitation Unit", "Available"),
        ("Vergence Rehabilitation Facility", "Available"),
        ("Visual Field Rehabilitation Lab", "Available"),
        ("Neuro-Visual Sentinel Monitoring Center", "Available")
    ]
    cursor.executemany(
        "INSERT INTO rehabilitation_resources (resource_name, availability) VALUES (?, ?)",
        resources
    )
    conn.commit()
    conn.close()

initialize_local_db()
print("Clinical Local Database Initialized Successfully.")

# ==========================================
# 7. AGENTIC EXPERT TOOL DEFINITIONS
# ==========================================
@tool
def ocular_knowledge_tool(question: str) -> str:
    """
    Use ONLY for technical questions about:
    - ocular rehabilitation, neuro-visual tracking, cranial nerve configurations.
    - gaze stabilization, saccadic movements, smooth pursuit configurations, blink reflexes.
    - Neuro-Visual Sentinel parameters, safety metrics, or structural clinical document contexts.
    """
    return rag_chain.invoke(question)

@tool
def check_availability_tool() -> str:
    """
    Use ONLY for checking available functional active ocular rehabilitation resources.
    """
    conn = sqlite3.connect("ocular_rehabilitation.db")
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, resource_name FROM rehabilitation_resources WHERE availability='Available'"
    )
    rows = cursor.fetchall()
    conn.close()
    if not rows:
        return "No ocular rehabilitation resources available currently."
    return "\n".join([f"Resource {r[0]}: {r[1]}" for r in rows])

@tool
def rehabilitation_resource_tool() -> str:
    """
    Returns a global master directory listing of all configured ocular rehabilitation infrastructure.
    """
    conn = sqlite3.connect("ocular_rehabilitation.db")
    cursor = conn.cursor()
    cursor.execute("SELECT resource_name FROM rehabilitation_resources")
    rows = cursor.fetchall()
    conn.close()
    if not rows:
        return "No ocular rehabilitation resources configured in data matrix registry."
    return "\n".join([f"Resource {i+1}: {r[0]}" for i, r in enumerate(rows)])

tools = [ocular_knowledge_tool, check_availability_tool, rehabilitation_resource_tool]

system_message = """
You are an AI-Powered Interactive Ocular Rehabilitation Assistant.

Available tools:
1. ocular_knowledge_tool -> technical ocular rehabilitation knowledge
2. check_availability_tool -> checks available ocular rehabilitation resources
3. rehabilitation_resource_tool -> lists available ocular rehabilitation resources

Always prioritize:
* Neuro-Visual Sentinel workflows
* Ocular rehabilitation guidance
* Cranial nerve rehabilitation
* Neuro-visual rehabilitation
* Ocular safety monitoring
* Tele-ocular rehabilitation support

Use ocular_knowledge_tool for technical rehabilitation questions.
Use availability tools for resource-related questions.
"""

# ==========================================
# 8. STATE ENGINE IMPLEMENTATION (LANGGRAPH)
# ==========================================
class State(TypedDict):
    messages: Annotated[list, add_messages]

def call_ocular_agent(state: State):
    messages = state['messages']
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=system_message)] + messages
    response_text = llm.invoke(messages)
    return {"messages": [AIMessage(content=response_text.content if hasattr(response_text, 'content') else str(response_text))]}

def custom_tools_condition(state: State):
    last_message = state['messages'][-1]
    text = last_message.content.lower()
    if "ocular_knowledge_tool" in text:
        return "tools"
    elif "check_availability_tool" in text:
        return "tools"
    elif "rehabilitation_resource_tool" in text:
        return "tools"
    return "__end__"

graph_builder = StateGraph(State)
graph_builder.add_node("agent", call_ocular_agent)
graph_builder.add_node("tools", ToolNode(tools=tools))

graph_builder.add_conditional_edges(
    "agent",
    custom_tools_condition,
    {
        "tools": "tools",
        "__end__": "__end__"
    }
)
graph_builder.add_edge("tools", "agent")
graph_builder.set_entry_point("agent")

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)
print("LangGraph Agent execution flow compiled cleanly.")

# ==========================================
# 9. DETERMINISTIC INTENT ROUTER GRID
# ==========================================
def healthcare_suite_router(user_query):
    query = user_query.lower()

    if "book" in query:
        return "Resource booking functionality is currently under development."

    elif "available" in query or "availability" in query:
        return check_availability_tool.invoke({})
        
    elif "resource" in query and ("list" in query or "all" in query):
        return rehabilitation_resource_tool.invoke({})

    elif any(
        x in query
        for x in [
            "ocular", "neuro-visual", "sentinel", "cranial nerve",
            "visual field", "saccadic", "smooth pursuit", "vergence", "stabilization"
        ]
    ):
        return ocular_knowledge_tool.invoke({"question": user_query})

    else:
        # Enforce secure fallback mapping straight to our technical safety prompt configuration
        return ocular_knowledge_tool.invoke({"question": user_query})

# ==========================================
# 10. GRADIO CLOUD USER INTERFACE RUNTIME
# ==========================================
def respond(message, history):
    try:
        user_message = message.get("content", "") if isinstance(message, dict) else str(message)
        result = healthcare_suite_router(user_message)
        return {
            "role": "assistant",
            "content": str(result)
        }
    except Exception as e:
        return {
            "role": "assistant",
            "content": f"Core Execution Exception: {str(e)}"
        }

description_html = """
<div style="text-align:center;">
    <h3>Built by Dr. Lakshmi Gandi</h3>
    <p>
    Interactive Ocular Rehabilitation Simulator<br>
    <b>Knowledge-Augmented Generation (KAG) + Unified Operational Matrix (YUOM)</b>
    </p>
</div>
"""

demo = gr.ChatInterface(
    fn=respond,
    type="messages",
    title="👁️ Interactive Ocular Rehabilitation Simulator",
    description=description_html,
    examples=[
        "How does the ocular rehabilitation simulator work?",
        "Explain the Neuro-Visual Sentinel workflow.",
        "Explain cranial nerve rehabilitation.",
        "What ocular rehabilitation resources are available?",
        "Explain gaze stabilization rehabilitation.",
        "Explain visual tracking rehabilitation.",
        "What safety mechanisms are described in the rehabilitation system?",
        "What is the future scope of the AI-powered ocular rehabilitation simulator?"
    ]
)

if __name__ == "__main__":
    # Launching execution runtime container binding setup configurations
    demo.launch()
