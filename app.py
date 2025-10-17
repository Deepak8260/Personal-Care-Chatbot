import streamlit as st
from services.llm_service import init_gemini_model
from services.db_service import init_database
from services.agent_service import create_agent

# --- Page Setup ---
st.set_page_config(page_title="Product Info Assistant", layout="centered")
st.title("💬 Personal Care Product Assistant")

# --- Initialize ---
@st.cache_resource
def setup_agent():
    global llm
    llm = init_gemini_model()
    db = init_database()
    agent = create_agent(llm, db)
    return agent

agent = setup_agent()

# --- Chat Interface ---
user_query = st.text_input("Ask about any product:")

if st.button("Ask") and user_query:
    with st.spinner("Analyzing your question..."):
        try:
            # Step 1: Get the raw response from the agent
            response = agent.invoke({"input": user_query})
            raw_output = response['output']

            # Step 2: Define cleaning instruction
            system_prompt = (
                "I will give you a query and the chat response in which there will be unwanted text in the response unrelated to the query. Your task is to clean the response and provide only the relevant answer to the query without any unrelated text and without any unwanted symbols like ** or * ."
)

            # Step 3: Use LLM to clean the response
            cleaner_llm = init_gemini_model()  # Reuse the same model for cleaning
            clean_prompt = f"{system_prompt}\n\nQuery: {user_query}\nResponse: {raw_output}\n\nCleaned:"
            cleaned_output = cleaner_llm.invoke(clean_prompt).content

            # Step 4: Display cleaned output
            st.success(cleaned_output)

        except Exception as e:
            st.error(f"Error: {e}")


st.markdown("---")
st.caption("Powered by LangChain + Gemini + Supabase")
