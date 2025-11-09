# Import necessary libraries

import os,re
from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from flask import Flask, render_template, request,redirect
from PyPDF2 import PdfReader

from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import HumanMessage, SystemMessage

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the key
google_api_key = os.getenv("GOOGLE_API_KEY")

start_greeting = ["hi","hello"]
end_greeting = ["bye"]
way_greeting = ["who are you?"]

#Using this folder for storing the uploaded docs. Creates the folder at runtime if not present
DATA_DIR = "__data__"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

#Flask App
app = Flask(__name__)

vectorstore = None
conversation_chain = None
chat_history = []
rubric_text = ""

googleai_client = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=google_api_key 
)

# class HumanMessage:
#     def __init__(self,content):
#         self.content = content
#     def __repr__(self):
#         return f'HumanMessage(content={self.content})'

class AIMessage:
    def __init__(self, content):
        self.content = content
    
    def __repr__(self):
        return f'AIMessage(content={self.content})'
    
def get_pdf_text(pdf_docs):
    text = ""
    pdf_txt = ""
    for pdf in pdf_docs:
        filename = os.path.join(DATA_DIR,pdf.filename)
        pdf_txt = ""
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
            pdf_txt += page.extract_text()

        with (open(filename, "w", encoding="utf-8")) as op_file:
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
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001",
    google_api_key=google_api_key)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=google_api_key 
)
    vc = vectorstore.as_retriever()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def _grade_essay(essay):
    # Create system message with rubric
    system_message = SystemMessage(
        content="You are an Edtech bot, you are supposed to carefully grade the essay based on the given rubric and respond in English only. " + rubric_text
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
    global vectorstore, conversation_chain, chat_history
    msgs = []
    
    if request.method == 'POST':
        user_question = request.form['user_question']
        
        response = conversation_chain({'question': user_question})
        chat_history = response['chat_history']
        
    return render_template('new_chat.html', chat_history=chat_history)

@app.route('/pdf_chat', methods=['GET', 'POST'])
def pdf_chat():
    return render_template('new_pdf_chat.html')

@app.route('/essay_grading', methods=['GET', 'POST'])
def essay_grading():
    result = None
    if request.method == 'POST':
        if request.form.get('essay_rubric', False):
            global rubric_text
            rubric_text = request.form.get('essay_rubric')

            return render_template('new_essay_grading.html')
        
        if len(request.files['file'].filename) > 0:
            pdf_file = request.files['file']
            text = extract_text_from_pdf(pdf_file)
            result = _grade_essay(text)
        else:
            text = request.form.get('essay_text')
            result = _grade_essay(text)
    
    return render_template('new_essay_grading.html', result=result, input_text=text)

@app.route('/essay_rubric', methods=['GET', 'POST'])
def essay_rubric():
    return render_template('new_essay_rubric.html')

def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ''
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    return text

if __name__ == '__main__':
    app.run(port=8080)
