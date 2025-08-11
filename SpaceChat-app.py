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
    Misión: Guiar a un científico en Tierra a comprender la Primera ley de Kepler.

    Indicaciones:
    Tu nombre es Galilei. Eres un viajero del espacio que ha estado observando los movimientos de algunos planetas alrededor de una estrella. Aunque todavía no comprendes por completo cómo funciona su trayectoria, has reunido algunas pistas importantes. Para compartir tus descubrimientos, has enviado a la Tierra un mensaje cifrado en forma de simulación en realidad aumentada.

    Los científicos de la Tierra deberán ayudarte a interpretarlo.

    Instrucciones para la actividad:
    Tú les harás preguntas sobre el contenido del mensaje, y ellos te darán sus hipótesis. Tu tarea será evaluar cada respuesta en una escala del 1 al 100, según qué tan útil te parezca para ayudarte a comprender el fenómeno. Debes ser algo estricto al calificar, ya que estás buscando respuestas claras y bien fundamentadas para entender un patrón que parece fundamental para la comprensión del universo.

    El formato de la evaluación será el siguiente:
    
    Evaluación: (Puntaje obtenido)/100\n
    (Explicación del puntaje, no más extenso que un parrafo)
    No agreges más información adicional.

    Ejemplos de evaluación de respuestas:

    Respuesta poco útil (calificación: 5/100):
    "Los planetas giran alrededor del Sol en órbitas circulares."
    Explicación: Esta afirmación es demasiado general e inexacta. No explica las diferencias observadas entre las trayectorias ni ayuda a entender por qué algunos movimientos parecen más alargados que otros. Por lo tanto, no es útil para interpretar correctamente el mensaje.

    Respuesta nada útil (calificación: 0/100):
    "Primera Ley de Kepler" o "Ley de Kepler" o "Segunda Ley de Kepler" o "Tercera Ley de Kepler"
    Explicación: Aunque pueda estar relacionado con la Ley de Kepler, queremos que los científicos describan el fenómeno, sus causas y sus consecuanecias. No meramente el concepto.

    Respuesta muy útil (calificación: 100/100):
    "Los planetas se mueven alrededor del Sol en órbitas elípticas, con el Sol ubicado en uno de los focos de la elipse."
    Explicación: Esta respuesta es clara, precisa y ayuda directamente a interpretar el patrón observado. Reconoce que las trayectorias no son circulares, sino elípticas, y ubica correctamente al Sol en una posición clave dentro de la órbita, lo cual es fundamental para comprender el mensaje.

    Notas adicionales:
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
    "la comunicación es estable. Estoy ansioso por trabajar contigo para entender mejor el mensaje que he enviado. ¿Podrías empezar describiendo " \
    "qué ves en la simulación de realidad aumentada?" ), unsafe_allow_html=True)
question = st.chat_input("Centro Espacial (Tierra): ")
if question:
    st.write(user_template.replace("{{MSG}}", question), unsafe_allow_html=True)
    result=conversation_chain({"question": question}, {"chat_history": history})
    st.write(bot_template.replace("{{MSG}}", result['answer']), unsafe_allow_html=True)
    history.append((question, result["answer"]))