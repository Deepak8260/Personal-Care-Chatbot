import streamlit as st
from services.llm_service import init_gemini_model
from services.db_service import (
    init_database,
    fetch_last_5_chats,
    store_chat
)
from services.agent_service import create_agent


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
                # Step 1️⃣: Fetch last 5 chats for context
                last_chats = fetch_last_5_chats()
                chat_context = ""
                for chat in last_chats:
                    chat_context += f"User: {chat['user_message']}\nAssistant: {chat['ai_response']}\n"

                # Step 2️⃣: Build contextual input
                contextual_prompt = (
                    f"Below is the chat history between the user and the assistant.\n"
                    f"Use it as context to answer the next question naturally and accurately.\n\n"
                    f"{chat_context}\n"
                    f"Current User Query: {user_query}\n"
                )

                # Step 3️⃣: Get raw response from the agent
                response = agent.invoke({"input": contextual_prompt})
                raw_output = response.get("output", "")

                # Step 4️⃣: Clean and synthesize final response
                system_prompt = (
                    "You are a helpful and detailed conversational assistant. I will provide you with a User Query and a Raw LLM Response. "
                    "Your task is to **fully synthesize a complete, detailed, and polite final response** that directly answers the User Query. "
                    "Use the Raw LLM Response as your primary source of fact, but **expand upon it** using clear, easy-to-understand language. "
                    "Your final output must be completely clean — remove any internal tags, system messages, or unwanted symbols like **, *, or #. "
                    "The final response should be a complete, grammatically correct, and human-readable paragraph."
                )

                full_prompt = f"User Query: {user_query}\n\nResponse: {raw_output}\n\n{system_prompt}"
                cleaned_output = llm.invoke(full_prompt).content

                # Step 5️⃣: Display response in UI
                st.markdown(cleaned_output)

                # Step 6️⃣: Save to database
                store_chat(user_query, cleaned_output)

                # Step 7️⃣: Append to session state
                st.session_state.messages.append(
                    {"role": "assistant", "content": cleaned_output}
                )

            except Exception as e:
                st.error(f"⚠️ Error: {e}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": "Sorry, something went wrong 😔."}
                )


# --- Footer ---
st.markdown("---")
st.caption("© 2025 Personal Care Assistant — All rights reserved.")
