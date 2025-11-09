import os
import re

# Vector store
from langchain_community.vectorstores import FAISS

# Google Generative AI
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# Flask
from flask import Flask, render_template, request, redirect

# PDF processing
from PyPDF2 import PdfReader

# Text splitters - CORRECTED
from langchain_text_splitters import CharacterTextSplitter

# Message schemas - CORRECTED
from langchain_core.messages import HumanMessage, SystemMessage

# Environment variables
from dotenv import load_dotenv

# Chains - CORRECTED
from langchain.chains import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain

# Prompts
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Chat history - ALL CORRECTED
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Load environment variables from .env file
load_dotenv()

# Get the API key
google_api_key = os.getenv("GOOGLE_API_KEY")

# Store for session histories
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def get_conversation_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        google_api_key=google_api_key 
    )
    
    retriever = vectorstore.as_retriever()
    
    # Prompt for contextualizing questions based on chat history
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question, "
                   "formulate a standalone question which can be understood "
                   "without the chat history. Do NOT answer the question, "
                   "just reformulate it if needed and otherwise return it as is."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # Create history-aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    
    # Prompt for answering questions
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant for question-answering tasks. "
                   "Use the following pieces of retrieved context to answer "
                   "the question. If you don't know the answer, say that you "
                   "don't know. Keep the answer concise.\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # Create question-answer chain
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # Create retrieval chain
    rag_chain = create_retrieval_chain(
        history_aware_retriever, question_answer_chain
    )
    
    # Wrap with message history
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    
    return conversational_rag_chain

# Using this folder for storing the uploaded docs. Creates the folder at runtime if not present
DATA_DIR = "__data__"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Flask App
app = Flask(__name__)

vectorstore = None
conversation_chain = None
chat_history = []
rubric_text = ""
current_session_id = "default_session"  # Add session tracking

googleai_client = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    google_api_key=google_api_key 
)

class AIMessage:
    def __init__(self, content):
        self.content = content
    
    def __repr__(self):
        return f'AIMessage(content={self.content})'
    
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        filename = os.path.join(DATA_DIR, pdf.filename)
        pdf_txt = ""
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            text += page_text
            pdf_txt += page_text

        with open(filename, "w", encoding="utf-8") as op_file:
            op_file.write(pdf_txt)

    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=google_api_key
    )
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def _grade_essay(essay):
    # Create system message with rubric
    system_message = SystemMessage(
        content=f"""You are an Edtech bot. Grade the essay based on the given rubric and respond in English only. 
        
        RUBRIC: {rubric_text}
        
        IMPORTANT: Do NOT repeat or restate the rubric in your response. Start directly with your assessment of the essay. Give assessment in less than 100 words."""
    )
    
    # Create user message with essay
    essay_content = "ESSAY : " + essay
    user_message = HumanMessage(content=essay_content)
    
    # Combine messages
    messages = [system_message, user_message]
    
    # Get response from Google AI
    response = googleai_client.invoke(messages)
    
    # Extract content
    data = response.content
    
    # Format for HTML display with proper styling
    # Convert markdown bold to HTML
    data = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', data)
    
    # Convert newlines to HTML breaks
    data = re.sub(r'\n', '<br>', data)
    
    # Add custom CSS styling
    formatted_output = f"""
    <div style="font-family: Arial, sans-serif; max-width: 800px; margin: 20px auto; padding: 20px; background: #f9f9f9; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
        {data}
    </div>
    """
    
    return formatted_output

@app.route('/')
def home():
    return render_template('new_home.html')

@app.route('/process', methods=['POST'])
def process_documents():
    global vectorstore, conversation_chain
    pdf_docs = request.files.getlist('pdf_docs')
    raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks)
    conversation_chain = get_conversation_chain(vectorstore)
    return redirect('/chat')

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    global vectorstore, conversation_chain, chat_history, current_session_id
    
    if request.method == 'POST':
        user_question = request.form['user_question']
        
        if conversation_chain:
            # Use the new invoke method with session config
            response = conversation_chain.invoke(
                {"input": user_question},
                config={"configurable": {"session_id": current_session_id}}
            )
            
            # Get the chat history from the session store
            session_history = get_session_history(current_session_id)
            chat_history = session_history.messages
        
    return render_template('new_chat.html', chat_history=chat_history)

@app.route('/pdf_chat', methods=['GET', 'POST'])
def pdf_chat():
    return render_template('new_pdf_chat.html')

@app.route('/essay_grading', methods=['GET', 'POST'])
def essay_grading():
    result = None
    text = ""
    
    if request.method == 'POST':
        if request.form.get('essay_rubric', False):
            global rubric_text
            rubric_text = request.form.get('essay_rubric')
            return render_template('new_essay_grading.html')
        
        if 'file' in request.files and len(request.files['file'].filename) > 0:
            pdf_file = request.files['file']
            text = extract_text_from_pdf(pdf_file)
            result = _grade_essay(text)
        else:
            text = request.form.get('essay_text', '')
            if text:
                result = _grade_essay(text)
    
    return render_template('new_essay_grading.html', result=result, input_text=text)

@app.route('/essay_rubric', methods=['GET', 'POST'])
def essay_rubric():
    return render_template('new_essay_rubric.html')

def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

if __name__ == '__main__':
    app.run(port=8080)