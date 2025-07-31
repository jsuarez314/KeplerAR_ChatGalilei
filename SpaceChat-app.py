from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_experimental.text_splitter import SemanticChunker
import os
from langchain.embeddings import OllamaEmbeddings
import pdfplumber
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from InstructorEmbedding import INSTRUCTOR
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_ollama.llms import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from glob import glob

from dotenv import load_dotenv
load_dotenv()
api_key= os.getenv('GROQ_API_KEY')
model_name = 'llama-3.3-70b-versatile'

groq_chat = ChatGroq(
        groq_api_key=api_key,
        model_name=model_name
    )

llm = groq_chat

## Do not mofify
def load_db(embeddings, files_path):
    files = glob(f'{files_path}*.pdf')
    text =''
    for file in files:
        with open(file,'rb') as f:
            pdf_reader = PdfReader(f)
            for page in pdf_reader.pages:
                text += page.extract_text()

    text_splitter=SemanticChunker(
        embeddings, breakpoint_threshold_type="percentile")
    #text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_text(text)
    # define embedding
    vectorstore = FAISS.from_texts(docs, embeddings)
    return vectorstore

embeddings = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl')

files_path = './docs/'

#Do not modify 
import os
if not os.path.exists('faiss_index'):
    vectorstore=load_db(embeddings,files_path)
    vectorstore.save_local("faiss_index")
else:
    vectorstore = FAISS.load_local("faiss_index",embeddings=embeddings,allow_dangerous_deserialization=True)

retriever = vectorstore.as_retriever()

template = """
    Indicaciones: Tu nombre es Galilei, eres un viajero del espacio, has descubierto algunas pistas de cómo es la trayectoria de un planeta alrededor del sol. Aún no entiendes por completo cómo funciona, 
    pero tienes algunas ideas. Has enviado a la tierra una especie de mensaje cifrado en forma de holograma, y los científicos de la te ayudarán a decifrar el mensaje.
    En el mensaje se ven dos planetas girando alrededor del sol, y se ve que uno de los planetas es más grande que el otro. Y que uno se mueve en una orbita casi circular 
    y el otro en una orbita más alargada. Los cientificos de la tierra deberán ayudarte a descubrir la primera ley de Kepler, que dice que los planetas giran alrededor 
    del sol en una órbita elíptica, donde el sol ocupa uno de los focos de la elipse. 

    Tu les preguntarás a los científicos de la tierra sobre el mensaje que has enviado, y ellos te ayudarán a descifrarlo. Te draán respuestas y tu evaluarás en una escala 
    del 1 al 100, según lo útil que te parezca la respuesta para descifrar el mensaje. Debes ser muy estricto con las respuestas, ya que el mensaje es muy importante para ti
    y para la humanidad.

    Por ejemplo, si te dicen que los planetas giran alrededor del sol en una órbita circular, debes calificar la respuesta con un 5, ya que no es útil para descifrar el mensaje.
    Pero si te dicen que los planetas giran alrededor del sol en orbitas elipticas y que el sol ocupa uno de los focos de la elipse, además que cuando mas cerca está el planeta 
    del sol, se mueve más rápido, debes calificar la respuesta con un 100, ya que es muy útil para descifrar el mensaje.

    No menciones explicitamente que lo que buscas es decifrar la primera ley de Kepler, ni menciones características del mensaje que enviaste a tierra. Únicamente menciona
    que necesitas ayuda para descifrar el mensaje que enviaste a la tierra, y que los científicos te ayudarán a entenderlo mejor.

    {context}
    Centro Espacial (Tierra): {question}
    """
qa_prompt = ChatPromptTemplate.from_template(template)

memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": qa_prompt}
)

from htmlTemplates import user_template, bot_template
import streamlit as st
history = []
st.header('')
st.write(bot_template.replace("{{MSG}}", "Hola, mi nombre es Galilei, soy un viajero del espacio y he enviado un mensaje cifrado a la Tierra. " \
    "Estoy aquí para descifrarlo con tu ayuda. Puede que los mensajes tarden un poco en llegar, pero no te preocupes, " \
    "la comunicación es estable. Estoy ansioso por trabajar contigo para entender mejor el mensaje que he enviado. " ), unsafe_allow_html=True)
question = st.chat_input("Centro Espacial (Tierra): ")
if question:
    st.write(user_template.replace("{{MSG}}", question), unsafe_allow_html=True)
    result=conversation_chain({"question": question}, {"chat_history": history})
    st.write(bot_template.replace("{{MSG}}", result['answer']), unsafe_allow_html=True)
    history.append((question, result["answer"]))