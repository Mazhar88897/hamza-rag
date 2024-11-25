# -*- coding: utf-8 -*-
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from groq import Groq
# Function to initialize Groq client
client = Groq(api_key="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")  # Replace with your actual API key

# Function to handle the language model response
def llm(question):
    """Simulate the HuggingFacePipeline using Groq API."""
    prompt = f"Question: {question}"
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama3-8b-8192",
    )
    return response.choices[0].message.content

# Page configuration
st.set_page_config(page_title="Xtreim", page_icon="X")



st.title("Xtreim")

# Initialize session state for messages and memory
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you ?"}]

if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )

# Display chat history
for message in st.session_state.messages:
    role = message["role"]
    with st.chat_message(role):
        st.markdown(message["content"])


# Function to load and process PDF
def load_and_process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    docs = text_splitter.split_documents(data)
    embeddings = HuggingFaceEmbeddings()
    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    return retriever

# Load PDF on first use
if "retriever" not in st.session_state:
    pdf_path = "bot.pdf"  # Replace with the path to your PDF file
    try:
        st.session_state.retriever = load_and_process_pdf(pdf_path)
    except Exception as e:
        st.error(f"Failed to load PDF: {e}")

retriever = st.session_state.get("retriever", None)

# User input
if user_question := st.chat_input("Ask a question about the PDF:"):
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # Generate response
    if retriever:
        with st.spinner("Thinking..."):
            context = retriever.invoke(user_question)
            context_text = "\n".join([doc.page_content for doc in context])
            prompt_template = """
            Respond confidently and briefly based on the given context and chat history.
            If outside the scope, respond with:
            "I'm sorry, I can't help with that. Is there anything else I can assist you with?"
            Dont mention according to context and chat history in response

            Chat History:
            {chat_history}

            Context:
            {context}

            Question: {question}

            Answer:
            """
            prompt = PromptTemplate(
                input_variables=["chat_history", "context", "question"],
                template=prompt_template,
            )
            chat_history = st.session_state.conversation_memory.chat_memory
            input_prompt = prompt.format(
                chat_history=chat_history, context=context_text, question=user_question
            )
            response = llm(input_prompt)

            # Update chat memory and session state
            st.session_state.conversation_memory.chat_memory.add_user_message(user_question)
            st.session_state.conversation_memory.chat_memory.add_ai_message(response) #type: ignore
            st.session_state.messages.append({"role": "assistant", "content": response}) #type: ignore
            with st.chat_message("assistant"):
                st.markdown(response)
    else:
        st.error("The retriever is not available. Please check your PDF setup.")

# Clear chat history button
