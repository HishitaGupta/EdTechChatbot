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

# Text splitters
from langchain_text_splitters import CharacterTextSplitter

# Message schemas
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Environment variables
from dotenv import load_dotenv

# Tools - UPDATED to modern approach
from langchain_core.tools import tool
from langchain_community.chat_message_histories import ChatMessageHistory

# Load environment variables from .env file
load_dotenv()

# Get the API key
google_api_key = os.getenv("GOOGLE_API_KEY")

# Store for session histories
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def get_conversation_agent(vectorstore):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        google_api_key=google_api_key,
        temperature=0
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # Create a retriever tool using @tool decorator
    @tool
    def search_documents(query: str) -> str:
        """Search for relevant information in the uploaded PDF documents.
        
        Args:
            query: The search query to find relevant document sections
            
        Returns:
            Relevant text from the documents
        """
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant documents found."
        return "\n\n".join([f"Document excerpt:\n{doc.page_content}" for doc in docs])
    
    tools = [search_documents]
    
    # Bind tools to LLM (modern approach)
    llm_with_tools = llm.bind_tools(tools)
    
    return llm_with_tools, tools

# Using this folder for storing the uploaded docs
DATA_DIR = "__data__"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Flask App
app = Flask(__name__)

vectorstore = None
conversation_agent = None
tools_list = None
chat_history = []
rubric_text = ""
current_session_id = "default_session"

googleai_client = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    google_api_key=google_api_key 
)
    
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
    system_message = SystemMessage(
        content=f"""You are an Edtech bot. Grade the essay based on the given rubric and respond in English only. 
        
        RUBRIC: {rubric_text}
        
        IMPORTANT: Do NOT repeat or restate the rubric in your response. Start directly with your assessment of the essay. Give assessment in less than 100 words."""
    )
    
    essay_content = "ESSAY : " + essay
    user_message = HumanMessage(content=essay_content)
    
    messages = [system_message, user_message]
    
    response = googleai_client.invoke(messages)
    
    data = response.content
    
    # Format for HTML display
    data = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', data)
    data = re.sub(r'\n', '<br>', data)
    
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
    global vectorstore, conversation_agent, tools_list
    pdf_docs = request.files.getlist('pdf_docs')
    raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks)
    conversation_agent, tools_list = get_conversation_agent(vectorstore)
    return redirect('/chat')

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    global vectorstore, conversation_agent, tools_list, chat_history, current_session_id
    
    if request.method == 'POST':
        user_question = request.form['user_question']
        
        if conversation_agent:
            try:
                # Get existing chat history
                session_history = get_session_history(current_session_id)
                
                # Create messages with history
                messages = []
                
                # Add system message
                messages.append(SystemMessage(
                    content="You are a helpful assistant that answers questions about uploaded PDF documents. Use the search_documents tool to find relevant information."
                ))
                
                # Add conversation history
                for msg in session_history.messages:
                    messages.append(msg)
                
                # Add current question
                messages.append(HumanMessage(content=user_question))
                
                # Invoke LLM with tools
                response = conversation_agent.invoke(messages)
                
                # Check if tool calls are needed
                if hasattr(response, 'tool_calls') and response.tool_calls:
                    # Execute tool calls
                    for tool_call in response.tool_calls:
                        tool_name = tool_call['name']
                        tool_args = tool_call['args']
                        
                        # Find and execute the tool
                        for tool in tools_list:
                            if tool.name == tool_name:
                                tool_result = tool.invoke(tool_args)
                                
                                # Add tool result to messages and get final response
                                from langchain_core.messages import ToolMessage
                                messages.append(response)
                                messages.append(ToolMessage(
                                    content=str(tool_result),
                                    tool_call_id=tool_call['id']
                                ))
                                
                                # Get final response
                                final_response = conversation_agent.invoke(messages)
                                response_content = final_response.content
                                break
                else:
                    response_content = response.content
                
                # Update chat history
                session_history.add_user_message(user_question)
                session_history.add_ai_message(response_content)
                
                chat_history = session_history.messages
                
            except Exception as e:
                print(f"Error in chat: {e}")
                import traceback
                traceback.print_exc()
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                session_history = get_session_history(current_session_id)
                session_history.add_user_message(user_question)
                session_history.add_ai_message(error_msg)
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
    app.run(port=8080, debug=True)