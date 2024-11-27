import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from groq import Groq
st.set_page_config(page_title="Xtream AI Assistant")

st.markdown(
    """
    <style>
    /* Hide Streamlit header and footer */
  
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* Remove padding and margin */
    .block-container {
        padding: 0 !important;
        margin: 0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
    /* Custom top bar */
    .top-bar {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 60px;
        background: linear-gradient(to left, #058cbd,black , black);
        color: white;
        display: flex;
        align-items: center;
        padding: 0 20px;
        font-size: 20px;
        font-weight: bold;
        z-index: 1000;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    }

    /* Icon styling */
    .top-bar img {
        height: 80px;
        margin-right: 15px;
        
    }

    /* Adjust the Streamlit app content to avoid overlap */
    .block-container {
        padding-top: 70px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# HTML for the top bar with icon
st.markdown(
    """
    <div class="top-bar">
        <img src="https://xtreim.com/wp-content/uploads/2023/03/Xtreim-insta-01_prev_ui-320x320.png" alt="Icon">
         AI Assistant
    </div>
    """,
    unsafe_allow_html=True
)



st.markdown('<style>.app-title {color: #0d47a1; font-size: 24px; font-weight: 600; font-family: "Segoe UI", "sans-serif"; } </style> <div class="app-title">Xtreim AI Assistant:</div>', unsafe_allow_html=True)

# Initialize Groq client
client = Groq(api_key="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")

def llm(question):
    prompt = f"Question: {question}"
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama3-8b-8192",
    )
    return response.choices[0].message.content

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"])

# PDF processing function
def load_and_process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    docs = text_splitter.split_documents(data)
    embeddings = HuggingFaceEmbeddings()
    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
    retriever =  vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    return retriever
# Load PDF
if "retriever" not in st.session_state:
    try:
        st.session_state.retriever = load_and_process_pdf("bot.pdf")
    except Exception as e:
        st.error(f"Failed to load PDF: {e}")

retriever = st.session_state.get("retriever", None)

# Chat input
if user_question := st.chat_input("Type your message here..."):
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_question)

    if retriever:
        with st.spinner("Processing your request..."):
            context = retriever.invoke(user_question)
            context_text = "\n".join([doc.page_content for doc in context])
            
            prompt_template = """
            Respond confidently and briefly based on the given context and chat history.
            If outside the scope, respond with:
            "I'm sorry, I can't help with that. Is there anything else I can assist you with?"
            Don't mention according to context and chat history in response

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
                chat_history=chat_history,
                context=context_text,
                question=user_question
            )
            
            response = llm(input_prompt)

            st.session_state.conversation_memory.chat_memory.add_user_message(user_question)
            st.session_state.conversation_memory.chat_memory.add_ai_message(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(response)
    else:
        st.error("The retriever is not available. Please check your PDF setup.")

st.markdown("</div>", unsafe_allow_html=True)