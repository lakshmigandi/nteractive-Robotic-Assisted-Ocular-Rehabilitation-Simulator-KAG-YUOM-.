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
print(f"Ext
