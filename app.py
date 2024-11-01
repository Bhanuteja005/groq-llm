import streamlit as st
import os

from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

def main():

    st.title("🌟 Groq Chat App 🌟")

    # Add information about Groq in the sidebar
    st.sidebar.title('About Groq')
    st.sidebar.info("Groq's Language Processing Unit (LPU) system aims to deliver lightning-fast inference speeds for Large Language Models (LLMs), surpassing other inference APIs such as those provided by OpenAI and Azure. Optimized for LLMs, Groq's LPU system provides ultra-low latency capabilities crucial for AI assistance technologies.")

    # Add customization options to the sidebar
    st.sidebar.title('Select an LLM')
    model = st.sidebar.selectbox(
        'Choose a model',
        ['mixtral-8x7b-32768', 'llama2-70b-4096']
    )
    conversational_memory_length = st.sidebar.slider('Conversational memory length:', 1, 10, value=5)

    memory = ConversationBufferWindowMemory(k=conversational_memory_length)

    st.markdown("### Ask a question:")
    user_question = st.text_area("", height=100)

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    else:
        for message in st.session_state.chat_history:
            memory.save_context({'input': message['human']}, {'output': message['AI']})

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("API key not found. Please set the GROQ_API_KEY environment variable.")
        return

    try:
        groq_chat = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=model
        )
    except ModuleNotFoundError:
        st.error("ChatGroq module not found. Please ensure it is installed.")
        return

    conversation = ConversationChain(
        llm=groq_chat,
        memory=memory
    )

    if st.button("Get Response"):
        if user_question:
            response = conversation(user_question)
            message = {'human': user_question, 'AI': response['response']}
            st.session_state.chat_history.append(message)
            
            st.markdown("#### You:")
            st.write(user_question)
            st.markdown("#### Chatbot:")
            st.write(response['response'])

            # Show balloons after getting response
            st.balloons()

    # Display chat history
    if st.session_state.chat_history:
        st.markdown("### Chat History")
        for message in st.session_state.chat_history:
            st.markdown(f"**You:** {message['human']}")
            st.markdown(f"**Chatbot:** {message['AI']}")
            st.markdown("---")

if __name__ == "__main__":
    main()