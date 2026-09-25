import streamlit as st
from rag import ask_question

st.set_page_config(
    page_title="Rag Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Rag Chatbot")
st.write("Ask questions about your uploaded documents.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question...")

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            try:
                answer, sources = ask_question(question)

                st.markdown(answer)
                

                

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as e:
                st.error(f"Error: {str(e)}")