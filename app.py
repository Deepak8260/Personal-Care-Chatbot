import streamlit as st
from services.llm_service import init_gemini_model
from services.db_service import init_database
from services.agent_service import create_agent
from langchain.schema import StrOutputParser  # ✅ Import string parser

# --- Page Setup ---
st.set_page_config(page_title="💬 Product Info Assistant", layout="centered")
st.title("🧴 Personal Care Product Chatbot")

# --- Initialize Agent ---
@st.cache_resource
def setup_agent():
    llm = init_gemini_model()
    db = init_database()
    agent = create_agent(llm, db)
    return agent, llm

agent, llm = setup_agent()
parser = StrOutputParser()  # ✅ Initialize parser

# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi 👋! I'm your personal care assistant. Ask me anything about our products!"}
    ]

# --- Display Chat History ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- User Input ---
user_query = st.chat_input("Type your message here...")

if user_query:
    # Display user message immediately
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing your question..."):
            try:
                # Step 1: Get raw response from the agent
                response = agent.invoke({"input": user_query})
                raw_output = response.get("output", "")

                # Step 2: Clean and parse output safely
                system_prompt = ("I will give you a query and the complete chat response in which there will be unwanted error and unwanted text in the response unrelated to the query. Your task is to provide only the relevant answer to the query without any unrelated text and without any unwanted text and without any unwanted symbols like ** or * .,"
                "The response should be a complete sentence in a proper meansingful way.")

                full_prompt = f"User Query: {user_query}\n\nResponse: {raw_output}\n\n{system_prompt}"
                cleaned_output = llm.invoke(full_prompt).content

                # ✅ Parse final text safely
                final_output = parser.invoke(cleaned_output)

                # Display and store response
                st.markdown(final_output)
                st.session_state.messages.append({"role": "assistant", "content": final_output})

            except Exception as e:
                st.error(f"⚠️ Error: {e}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": "Sorry, something went wrong 😔."}
                )

# --- Footer ---
st.markdown("---")
st.caption("Powered by LangChain + Gemini + Supabase")
