import streamlit as st
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from PyPDF2 import PdfReader

# Initialize the Gemini chatbot
gemini_chat = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)

# Streamlit UI
st.set_page_config(page_title="Chatbot for help")
st.title("Let's chat with DOC'S")

# Initialize conversation flow messages in the Streamlit session state
if 'flowmessages' not in st.session_state:
    st.session_state['flowmessages'] = [
        SystemMessage(content="You are chatting with an AI assistant")
    ]

# Function to get response from Gemini chatbot
def get_gemini_response(user_message):
    st.session_state['flowmessages'].append(HumanMessage(content=user_message))
    answer = gemini_chat(st.session_state['flowmessages'])
    st.session_state['flowmessages'].append(AIMessage(content=answer.content))
    return answer.content

# Function to extract text from PDF file
def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return None

# Function to display chat history
def display_chat_history():
    for message in st.session_state['flowmessages']:
        if isinstance(message, HumanMessage):
            st.text_area("You:", value=message.content, height=100)
        elif isinstance(message, AIMessage):
            st.text_area("Bot:", value=message.content, height=500)

# Main function
def main():
    # File uploader for PDF documents
    st.sidebar.title("Upload PDF Document")
    pdf_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

    if pdf_file:
        text = extract_text_from_pdf(pdf_file)
        if text:
            user_input = st.text_input("You:")
            if st.button("Send"):
                response = get_gemini_response(user_input)
                display_chat_history()
        else:
            st.error("Failed to extract text from PDF file.")

if __name__ == "__main__":
    main()
