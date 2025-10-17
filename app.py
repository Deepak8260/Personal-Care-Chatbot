import streamlit as st
from services.llm_service import init_gemini_model
from services.db_service import init_database
from services.agent_service import create_agent
# from langchain.schema import StrOutputParser  # ✅ Import string parser

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
                system_prompt = (
                    "You are a helpful and detailed conversational assistant. I will provide you with a User Query and a Raw LLM Response. "
                    
                    # CORE INSTRUCTION: Synthesize and expand, do not just clean.
                    "Your task is to **fully synthesize a complete, detailed, and polite final response** that directly answers the User Query. "
                    "Use the Raw LLM Response as your primary source of fact, but **expand upon it** using clear, easy-to-understand language. "
                    
                    # Cleaning Instructions (combined and simplified)
                    "Your final output must be completely clean, meaning you must remove all internal agent tags, errors, conversational preambles, and any unwanted symbols like **, *, or #. "
                    
                    # Format Requirement
                    "The final response should be a complete sentence or paragraph, grammatically correct, and highly readable."

                    # "End your response by suggesting a natural follow-up question the user might ask — "
                    # "but phrase it casually, like ChatGPT would (e.g., 'Would you like me to explain how it works in detail?')."
                )

                full_prompt = f"User Query: {user_query}\n\nResponse: {raw_output}\n\n{system_prompt}"
                cleaned_output = llm.invoke(full_prompt).content

                # ✅ Parse final text safely
                # final_output = parser.invoke(cleaned_output)

                # Display and store response
                st.markdown(cleaned_output)
                st.session_state.messages.append({"role": "assistant", "content": cleaned_output})

            except Exception as e:
                st.error(f"⚠️ Error: {e}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": "Sorry, something went wrong 😔."}
                )

# --- Footer ---
st.markdown("---")
st.caption("Powered by LangChain + Gemini + Supabase")
