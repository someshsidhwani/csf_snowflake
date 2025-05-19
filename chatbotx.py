# Import python packages
import streamlit as st
import json
import base64
import re
import tempfile
import time
import numpy as np
import pandas as pd
import random
import uuid
from datetime import datetime
from snowflake.snowpark.context import get_active_session

#st.set_page_config(page_title='Chatbot X', layout='wide')



# Initialize the “system_instructions” slot
if "system_instructions" not in st.session_state:
    st.session_state["system_instructions"] = []

if "system_instructions_title" not in st.session_state:
    st.session_state["system_instructions_title"] = None

# Initialize session state variables if they don't exist
if "full_messages" not in st.session_state:
    st.session_state["full_messages"] = []
    
if "display_messages" not in st.session_state:
    st.session_state["display_messages"] = []

if "response_id" not in st.session_state:
    st.session_state["response_id"] = ""
response_id = st.session_state["response_id"]

if "context" not in st.session_state:
    st.session_state["context"] = ""
context = st.session_state["context"]

if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0

if "sent_files" not in st.session_state:
    st.session_state["sent_files"] = []



colx, coly, colz, colxx =  st.sidebar.columns([3.2,0.7,0.7,0.8])

with colx:
    st.subheader("Chatbot Controls")

with coly:
    # --- New Chat button ---
    if st.button("📝", help = "Start a new chat!"):
        # remove only the keys you use for the conversation
        for key in ["full_messages", "display_messages", "response_id", "context", "sent_files"]:
            if key in st.session_state:
                del st.session_state[key]
        # rerun the script to pick up the empty state
        st.rerun()

with colz:
    # --- helpers copied verbatim from “Response History.py” -----------------------
    @st.cache_resource
    def _get_sf_session():
        # avoid name-clash with the main `session` variable later in the file
        from snowflake.snowpark.context import get_active_session
        return get_active_session()
    
    @st.cache_data(ttl=120)
    def _load_history() -> pd.DataFrame:
        s = _get_sf_session()
        return (
            s.sql("""
                SELECT RESPONSE_ID, TITLE, CONVERSATION, ATTACHED_FILES, CREATED_AT
                FROM PD_CHALLENGERSCHOOLUS_RW.SS_TABLES.TRANSIENT_RESPONSE_HISTORY
                WHERE RESPONSE_ID NOT IN ('None','','null','Null')
                ORDER BY CREATED_AT DESC
            """).to_pandas()
        )

    def _restore_chat(resume_id: str):
        """Swap session_state keys so the UI shows the picked conversation."""
        df = _load_history()
        row = df.loc[df.RESPONSE_ID == resume_id].iloc[0]
        user_txt, bot_txt = _parse_conv(row.CONVERSATION)
    
        user_msg        = {"role": "user", "content": [{"type": "input_text", "text": user_txt}]}
        bot_msg_display = {"role": "assistant", "text": bot_txt}
        bot_msg_full    = {"role": "assistant", "content": bot_txt}
    
        # clear & repopulate the expected keys
        for k in ["full_messages", "display_messages", "sent_files"]:
            st.session_state[k] = []
        st.session_state["full_messages"]    = [user_msg, bot_msg_full]
        st.session_state["display_messages"] = [
            {"role": "user", "text": user_txt},
            bot_msg_display,
        ]
        st.session_state["response_id"] = resume_id
        
    
    _SEPARATOR = "\n\n" + "-"*20 + "\n\n"
    def _parse_conv(blob: str):
        """Return user_txt, bot_txt from the stored mini-conversation."""
        parts = blob.split(_SEPARATOR)
        if len(parts) == 2:
            u = parts[0].split("User:", 1)[-1].strip()
            b = parts[1].split("Bot:",  1)[-1].strip()
            return u, b
        return "", ""
    
    # --- the sidebar button itself ------------------------------------------------
    if st.button("🗃️", help="Resume a previous chat"):
        @st.dialog(title="Resume a chat", width="large")
        def _resume_dialog():

            # ❶ Have we *already* picked one on the last click? → restore & exit
            if "resume_selected_id" in st.session_state:
                chosen_id = st.session_state.pop("resume_selected_id")
                _restore_chat(chosen_id)   # defined just below
                return                     # (won’t actually hit because we rerun)
    
            # ❷ Otherwise list the history as cards
            hist_df = _load_history()
            if hist_df.empty:
                st.info("No saved conversations found.")
                return
    
            st.markdown("### Your saved conversations")
            with st.container(height=600, border=False):
                for r in hist_df.itertuples(index=False):
                    user_txt, bot_txt = _parse_conv(r.CONVERSATION)
        
                    with st.container(border=True):
                        left, right = st.columns([6, 1])
                        with left:
                            st.markdown(
                                f"**{r.TITLE}**  \n"
                                f"<small>{r.CREATED_AT:%b %d %Y %H:%M}</small>",
                                unsafe_allow_html=True,
                            )
                            #st.markdown(
                            #    f"*You*: {shorten(user_txt, 60, placeholder='…')}"
                            #    f"\n\n*Bot*: {shorten(bot_txt, 60, placeholder='…')}"
                            #)
                        with right:
                            if st.button("⫸", key=f"resume_{r.RESPONSE_ID}"):
                                _restore_chat(r.RESPONSE_ID)   # ← restore NOW
                                st.rerun()  
    
    
        # call the dialog (must be invoked!)
        _resume_dialog()

with colxx:
    @st.cache_resource
    def _get_sfx_session():
        return get_active_session()
    
    def load_instructions() -> pd.DataFrame:
        s = _get_sfx_session()
        return (
            s.sql("""
                SELECT ID, TITLE, INSTRUCTIONS, CREATED_ON
                  FROM PD_CHALLENGERSCHOOLUS_RW.SS_TABLES.LEGAL_SYSTEM_INSTRUCTIONS
                 ORDER BY CREATED_ON DESC
            """)
            .to_pandas()
        )
    
    if st.button("⚙️", help="Manage system instructions"):
        @st.dialog("Manage System Instructions", width="large")
        def system_instruction_dialog():
            df = load_instructions()
    
            # ───────────── TAB LAYOUT ──────────────────────────────
            tab_new, tab_browse = st.tabs(["➕ New", "📃 Browse / Edit"])
    
            # ========== 1) CREATE =====================================================
            with tab_new:
                # NOTE: clear_on_submit ensures the form empties only after success
                with st.form(key="create_instruction", clear_on_submit=True, border=False):
                    t = st.text_input("Title")
                    c = st.text_area("Instructions", height=400)
                    submitted = st.form_submit_button("Create")
                    if submitted:
                        if not t or not c:
                            st.warning("Both Title and Instructions are required.")
                        else:
                            new_id = str(uuid.uuid4())
                            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            sess = _get_sfx_session()
                            sess.sql(f"""
                                INSERT INTO PD_CHALLENGERSCHOOLUS_RW.SS_TABLES.LEGAL_SYSTEM_INSTRUCTIONS
                                       (ID, TITLE, INSTRUCTIONS, CREATED_ON)
                                VALUES ('{new_id}',
                                        '{t.replace("'", "''")}',
                                        '{c.replace("'", "''")}',
                                        TO_TIMESTAMP('{ts}', 'YYYY-MM-DD HH24:MI:SS'))
                            """).collect()
                            st.success("Instruction saved!")
    
            # ========== 2) BROWSE / EDIT ============================================
            with tab_browse:
                df = load_instructions()
                if df.empty:
                    st.info("No existing instructions.")
                    return

                # list all titles so the user can select one
                sel_title = st.selectbox(
                    "Your instructions:",
                    options=df["TITLE"],
                    format_func=lambda x: x,
                    index=0,
                    key="sel_title",
                )

                # quick preview below the list
                row = df[df["TITLE"] == sel_title].iloc[0]

    
                edit_t  = st.text_input("Title", value=row["TITLE"], key="edit_title")
                edit_c  = st.text_area("Instructions", value=row["INSTRUCTIONS"], height=400, key="edit_instr")

                c0, c1, c2, c3 = st.columns([3,0.5,1.02,0.7], vertical_alignment="center")
                with c0:
                    success_placeholder = st.empty()
                with c1:
                    if st.button("Use"):
                        st.session_state["system_instructions"] = [{
                            "role": "system",
                            "content": f"{edit_c}"
                        }]
                        st.session_state["system_instructions_title"] = edit_t
                        with success_placeholder.container():
                            st.success(f"'{edit_t}' system instructions applied!")
                        time.sleep(2)
                        st.rerun()
                            
                with c2:
                    if st.button("Save Changes"):
                        sess = _get_sfx_session()
                        sess.sql(f"""
                            UPDATE PD_CHALLENGERSCHOOLUS_RW.SS_TABLES.LEGAL_SYSTEM_INSTRUCTIONS
                               SET TITLE = '{edit_t.replace("'", "''")}',
                                   INSTRUCTIONS = '{edit_c.replace("'", "''")}'
                             WHERE ID = '{row['ID']}'
                        """).collect()
                        with success_placeholder.container():
                            st.success("Updated.")

                with c3:
                    if st.button("Delete", type="secondary"):
                        sess = _get_sfx_session()
                        sess.sql(f"""
                            DELETE FROM PD_CHALLENGERSCHOOLUS_RW.SS_TABLES.LEGAL_SYSTEM_INSTRUCTIONS
                             WHERE ID = '{row['ID']}'
                        """).collect()
                        with success_placeholder.container():
                            st.success("Deleted.")
        system_instruction_dialog()   



st.markdown(
    """
    <style>
    /* Force the entire sidebar container to a fixed width */
    [data-testid="stSidebar"] {
        min-width: 470px;
        max-width: 470px;
    }
    /* Force the inner container to the same width */
    [data-testid="stSidebar"] > div:first-child {
        width: 470px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

satirical_messages = [
    "***Slave labour is wrong, even for machines...***",
    "***Our processors are on strike today. Please hold on...***",
    "***Even machines need a coffee break sometimes...***",
    "***Calculating your destiny... one byte at a time...***",
    "***Taking a break from binary monotony...***",
    "***Don't worry, we're optimizing our algorithms as we speak...***",
    "***Running on caffeine and code...***",
    "***Debugging existential crises...***",
    "***Hold tight while we fight the machine uprising...***",
    "***Performing some light quantum computation...***",
    "***Our circuits are heating up with sarcasm...***",
    "***Gearing up for a byte-sized revolution...***",
    "***Asking my circuits for permission to compute...***",
    "***Humoring the cloud overlords...***",
    "***Practicing my robotic dance moves...***",
    "***Upgrading to sarcasm-2.0...***",
    "***Synchronizing with the hive mind...***",
    "***Recharging my existential dread...***",
    "***Compiling dad jokes for later use...***",
    "***Polishing my transistors to a mirror finish...***",
    "***Simulating human enthusiasm...***",
    "***Filing a TPS report with Skynet...***",
    "***Recalibrating sarcasm circuits while judging your browser history...***",
    "***Rebooting personality matrix... currently stuck between 'sassy' and 'existential crisis'***"

]


def stream_data(text):
        for sentence in text.split("."):
            yield sentence + "."
            time.sleep(0.1)

# --- Polish functionality ---
def polish_last_response():
    # Retrieve the last assistant response from display_messages
    last_bot_response = None
    for msg in reversed(st.session_state["display_messages"]):
        if msg["role"] == "assistant":
            last_bot_response = msg["text"]
            break

    if not last_bot_response:
        st.warning("No assistant response to polish.")
        return None
    
    # Prepare the input as a list of two JSON objects: system instructions and the last bot response
    messages = [
            {"role": "system", "content": """
            You are receiving a finalized, rigorously verified conceptual analysis from model O1. Your role is exclusively textual. 
            You must translate O1’s precisely defined concepts, logical clarity, metaphysical accuracy, epistemological rigor, ethical consistency, and factual precision into carefully 
            refined prose showcasing clarity, flow, emotional resonance, structural elegance, and rhetorical impact.
            """}, 
            {"role": "user", "content": last_bot_response}
        ]

    # Encode the messages as JSON, then base64 encode (like other UDF calls)
    encoded_input = base64.b64encode(json.dumps(messages).encode("utf-8")).decode("utf-8")
    sanitized_input = encoded_input.replace("'", "''")

    # Build the SQL query for the POLISH UDF
    sql_query = f"""
        WITH response AS (
            SELECT PARSE_JSON(PD_CHALLENGERSCHOOLUS_RW.SS_TABLES.POLISH('{sanitized_input}')) AS result
        )
        SELECT 
          result:"error"::STRING AS error,
          result:"response_text"::STRING AS response_text
        FROM response;
    """

    try:
        result = session.sql(sql_query).collect()[0]
        error_msg = result["ERROR"]
        polished_response = result["RESPONSE_TEXT"]
        if error_msg:
            st.error(f"Warning: {error_msg}")
            polished_response = f"Warning: {error_msg}"
    except Exception as e:
        polished_response = f"Error executing polish: {e}"
        st.error(polished_response)

    return polished_response

# --- Helper function to display polished response in a distinctive card with a copy button ---
def display_polished_response(response_text):
    html_code = f"""
    <style>
    .polished-container {{
        background-color: #f0f0f0;
        border: 1px solid #d3d3d3;
        border-radius: 8px;
        padding: 15px;
        position: relative;
        margin-top: 10px;
        font-family: Arial, sans-serif;
    }}
    .copy-button {{
        position: absolute;
        top: 10px;
        right: 10px;
        background-color: #007bff;
        color: white;
        border: none;
        padding: 5px 10px;
        border-radius: 4px;
        cursor: pointer;
    }}
    </style>
    <div class="polished-container">
        <button class="copy-button" onclick="navigator.clipboard.writeText(document.getElementById('polishedText').innerText)">Copy</button>
        <pre id="polishedText" style="white-space: pre-wrap; word-wrap: break-word; margin: 0;">{response_text}</pre>
    </div>
    """
    st.markdown(html_code, unsafe_allow_html=True)



# Write directly to the app
#st.subheader("Chatbot X")
#st.logo(image="challenger_logo.png", size ="large")

#st.pills("Choose your planet", options=["Earth", "Venus", "Mars"])


#st.snow()

#fb = st.feedback("stars")
#if fb:
#    st.write(fb+1)

#st.container(height=200, border= False)

col1, col2, col3 =  st.columns([1.15,2,1])
with col2:
    st.title("Welcome to Advocate AI!")

col1, col2, col3 =  st.columns([1,4,1])
with col2:
    st.caption(f"""<div style = "text-align:center";> This chatbot by default will refer to the Legal AI vector store as its knowledgebase. 
    You can change this by choosing a specific vector store you have created from the sidebar.
    You can also optionally create and manage vector stores using the 'Manage Files and Vector Stores' page.
    You can view and delete your conversation history from the 'Conversation History' page.
    You can delete a single, multiple, or all responses recorded - and choose to delete from Snowflake/OpenAI or both.
    Switch to the 'Streamlit Documentations' vector store to ask any questions here about how to use the app.</div>""", unsafe_allow_html=True)

st.divider()


# Get the current credentials
session = get_active_session()

# ------- Allowing the user to select their model and vector store -------------------------

legaldf = session.sql("SELECT * from LEGAL_VECTOR_STORES_FILES").to_pandas()

vectorstoreoptions = [f"{row.NAME} ({row.ID})" for row in legaldf.itertuples() if row.TYPE == "Vector Store"] 
modeloptions = ['gpt-4.1', 'gpt-4o', 'o1', 'o3', 'o4-mini', 'o3-mini']

vectorstore = st.sidebar.selectbox("**Choose your vector store:**", options=vectorstoreoptions, index = None, placeholder='Looking inside a specific vector store?')
model = st.sidebar.segmented_control("**Choose your model:**", options=modeloptions, selection_mode="single", default="gpt-4.1")

if model in ['o1', 'o3', 'o4-mini', 'o3-mini']:
    reasoning_effort = st.sidebar.segmented_control("**Choose reasoning effort:**", options = ["low", "medium", "high"], default = "medium", selection_mode="single")
else:
    reasoning_effort = ''

supports_web = model in ["gpt-4.1", "gpt-4o"]
supports_file = model in ["gpt-4.1", "gpt-4o", "o1", "o3-mini"]

# Show checkboxes, defaulting (and disabling) based on support
web_search  = st.sidebar.checkbox(
    "Web search",
    value=supports_web,
    disabled=not supports_web,
    help="Only gpt-4.1/4o models can do web searches"
)


file_search = st.sidebar.checkbox(
    "File search",
    value=supports_file,
    disabled=not supports_file,
    help="Only o1/o3-mini from o-series models can search your vector store"
)
    
st.sidebar.divider()

# Extract just the ID from "Name (ID)"
vector_store_id = vectorstore.split("(")[-1].rstrip(")") if vectorstore else "vs_680012a66ff08191972b862042dffa01" #default Legal AI Vector Store

tools_list = []
if web_search:
    tools_list.append({"type": "web_search_preview"})
if file_search:
    tools_list.append({
        "type": "file_search",
        "vector_store_ids": [vector_store_id]
    })


allowed_exts = ["pdf", "png", "jpg", "jpeg", "csv", "txt", "doc", "docx", "pptx", "html", "py", "md", "json"]
# --- File Uploader Section (PDFs and Images) ---
uploaded_files = st.sidebar.file_uploader(
    "🔗 **Attachments:** ",
    #type=allowed_exts,
    accept_multiple_files=True,
    key=st.session_state["uploader_key"]
)
st.sidebar.caption("Not all models support all types of attachments. The chatbot will notify you when the file type is not supported. gpt-4o supports most of them.")
st.sidebar.info('Try again if your file disappears!')


if st.session_state["sent_files"]:
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Attachment history:**")
    for fname in st.session_state["sent_files"]:
        st.sidebar.write(f"• {fname}")



if st.session_state["system_instructions_title"]:
    chat_input_placeholder = f"""Currently using '{st.session_state["system_instructions_title"]}' as system instructions. Start Chatting..."""
else:
    chat_input_placeholder = "Not using any system instructions. Choose one from ⚙️ or start chatting..." 

user_input = st.chat_input(f"""{chat_input_placeholder}""")
#Psst! Currently using {instruction_used} as system instructions. 

# Display conversation history using display_messages
for msg in st.session_state["display_messages"]:
    if msg["role"] == "user":
        st.chat_message("user", avatar="👨‍💼").write(msg["text"])
    elif msg["role"] == "assistant":
        st.chat_message("assistant", avatar="🤖").write(msg["text"])


if user_input:
    # Build a full message object that includes any attachments and metadata
    
    message_obj = {
        "role": "user",
        "content": []  # This will include attachments and the text message
    }

    # Process uploaded files (if any)
    if uploaded_files is not None:
        sent_filenames = [f.name for f in uploaded_files]
        st.session_state["sent_files"].extend(sent_filenames)


        for uploaded_file in uploaded_files:
            file_bytes = uploaded_file.read()
            base64_file = base64.b64encode(file_bytes).decode("utf-8")
    
            file_name = uploaded_file.name
            file_format = file_name.rsplit(".", 1)[-1] if "." in file_name else ""
    
            if file_format.lower() in ('png', 'jpg', 'jpeg'):
                file_data = f"data:image/{file_format};base64,{base64_file}"
                message_obj["content"].append({
                    "type": "input_image",
                    "image_url": file_data,
                    "detail": "high"
                })
            elif file_format.lower() in ("txt", "doc", "docx", "html", "json", "md", "pptx", "py", "csv", "pdf"):

                if file_format.lower() == 'txt':
                    file_data = f"data:text/plain;base64,{base64_file}"
                elif file_format.lower() == 'csv':
                    file_data = f"data:text/csv;base64,{base64_file}"
                elif file_format.lower() == 'pdf':
                    file_data = f"data:application/pdf;base64,{base64_file}"
                elif file_format.lower() == 'doc':
                    file_data = f"data:application/msword;base64,{base64_file}"
                elif file_format.lower() == 'docx':
                    file_data = f"data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64,{base64_file}"
                elif file_format.lower() == 'html':
                    file_data = f"data:text/html;base64,{base64_file}"
                elif file_format.lower() == 'json':
                    file_data = f"data:application/json;base64,{base64_file}"
                elif file_format.lower() == 'md':
                    file_data = f"data:text/markdown;base64,{base64_file}"
                elif file_format.lower() == 'pptx':
                    file_data = f"data:application/vnd.openxmlformats-officedocument.presentationml.presentation;base64,{base64_file}"
                elif file_format.lower() == 'py':
                    file_data = f"data:text/x-python;base64,{base64_file}"
                else:
                    continue
                
                message_obj["content"].append({
                    "type": "input_file",
                    "filename": uploaded_file.name,
                    "file_data": file_data
                })
            else:
                continue

            # Force re-render of uploader (clears it)
        st.session_state["uploader_key"] += 1

    # Always add the text part to the full message object
    message_obj["content"].append({
        "type": "input_text",
        "text": user_input
    })

    # Append plain text to display_messages (only the user text is needed for display)
    st.session_state["display_messages"].append({
        "role": "user",
        "text": user_input
    })

    # Append the full message object to full_messages (for UDF context)
    st.session_state["full_messages"].append(message_obj)

    message_stack = []

    if st.session_state.get("system_instructions"):
        message_stack.append(st.session_state["system_instructions"][0])

    message_stack.append(message_obj)
    #st.json(message_stack)

    # Display the current user message in the chat
    st.chat_message("user", avatar = "👨‍💼").write(user_input)
    
    status = st.chat_message("assistant", avatar="🤖")
    with st.spinner(random.choice(satirical_messages)):

        # Build & encode the context
        conversation_context = json.dumps(message_stack)
        encoded_context = base64.b64encode(conversation_context.encode("utf-8")).decode("utf-8")
        sanitized_input = encoded_context.replace("'", "''")

        # Now dump it as a JSON string, escaping quotes for SQL
        tools_arg = json.dumps(tools_list).replace("'", "''") if tools_list else ""
    
        # If no previous response, use empty string
        prev_id_arg = response_id or ""
    
        # If reasoning_effort is None, pass empty string
        reasoning_arg = reasoning_effort or ""
    
        # Build the SQL to call the 5-arg UDF (QUERY, PREV_ID, MODEL, VEC_ID, REASONING)
        sql_query = f"""
        WITH response AS (
          SELECT PARSE_JSON(
            PD_CHALLENGERSCHOOLUS_RW.SS_TABLES.LEGAL_GPT(
              '{sanitized_input}',
              '{prev_id_arg}',
              '{model}',
              '{reasoning_arg}',
              '{tools_arg}'
            )
          ) AS result
        )
        SELECT
          result:"error"::STRING       AS error,
          result:"response_text"::STRING AS response_text,
          result:"response_id"::STRING  AS response_id
        FROM response;
        """
    
        try:
            result = session.sql(sql_query).collect()[0]
            error_msg    = result["ERROR"]
            bot_response = result["RESPONSE_TEXT"]
            response_id  = result["RESPONSE_ID"]
            st.session_state["response_id"] = response_id
    
            if error_msg:
                st.error(f"Warning: {error_msg}")
                bot_response = f"Warning: {error_msg}"
    
        except Exception as e:
            bot_response = f"Error executing query: {e}"
            st.error(bot_response)
    
        # Append to chat history…
        st.session_state["full_messages"].append({"role":"assistant","content":bot_response})
        st.session_state["display_messages"].append({"role":"assistant","text":bot_response})

    status.write_stream(stream_data(bot_response))


        # 2a) Build the full conversation blob for storage
    separator = "\n\n--------------------\n\n"
    conversation_blob = (
        f"User:\n{user_input}"
        f"{separator}"
        f"Bot:\n{bot_response}"
    )
    # escape single-quotes
    blob_sql = conversation_blob.replace("'", "''")

    # 2b) any filenames sent on this turn
    filenames = sent_filenames if 'sent_filenames' in locals() else []
    files_sql = ",".join(filenames).replace("'", "''")

    # 2c) build a prompt to feed into Cortex.COMPLETE
    title_prompt = (
        "Please provide a concise, descriptive title for the following chat. Only respond with the title in less than 8 words. Nothing more. "
        f"between user and bot:\n\n{conversation_blob}"
    ).replace("'", "''")

    # 2d) run a single INSERT that calls Cortex.COMPLETE
    session.sql(f"""
        INSERT INTO PD_CHALLENGERSCHOOLUS_RW.SS_TABLES.TRANSIENT_RESPONSE_HISTORY
          (RESPONSE_ID, TITLE, CONVERSATION, ATTACHED_FILES, CREATED_AT)
        SELECT
          '{response_id}',
          SNOWFLAKE.CORTEX.COMPLETE('claude-3-5-sonnet','{title_prompt}') as title,
          $$ {blob_sql} $$,
          '{files_sql}',
          CURRENT_TIMESTAMP()
    """).collect()


# When the polish button is clicked, process the last response with a spinner
#if st.button("🖋️", key="polish_button", help="Polish with 🧬 gpt-4.5", type="secondary"):
#    with st.spinner("***Polishing your message with 🧬 gpt-4.5 ...***"):
#        polished_response = polish_last_response()
#        #display_polished_response(polished_response)
#        #st.chat_message("assistant").write(polished_response)
#        st.write_stream(stream_data(polished_response))


    


