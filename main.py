import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from PyPDF2 import PdfReader

# Initialize the Gemini chatbot with the necessary parameters
gemini_chat = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2)

# Custom exception for handling errors related to Google Generative AI
class ChatGoogleGenerativeAIError(Exception):
    """Custom exception for handling errors related to Google Generative AI."""
    pass

# Streamlit UI
st.set_page_config(page_title="Chatbot for help")
st.title("Let's chat with DOC'S")

# Initialize conversation flow messages in the Streamlit session state
if 'flowmessages' not in st.session_state:
    st.session_state['flowmessages'] = []

# Function to get response from Gemini chatbot
def get_gemini_response(user_message):
    try:
        # Append user message to conversation history
        st.session_state['flowmessages'].append(user_message)
        # Call the chat model with the updated conversation history
        answer = gemini_chat.predict(st.session_state['flowmessages'])
        # Append AI response to conversation history
        st.session_state['flowmessages'].append(answer)
        # Return AI response
        return answer
    except Exception as e:
        raise ChatGoogleGenerativeAIError(f"Error occurred: {e}")

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
        st.text_area("Chat History:", value=message, height=200)

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
                try:
                    response = get_gemini_response(user_input)
                    display_chat_history()
                except ChatGoogleGenerativeAIError as e:
                    st.error(e)
        else:
            st.error("Failed to extract text from PDF file.")

if __name__ == "__main__":
    main()
