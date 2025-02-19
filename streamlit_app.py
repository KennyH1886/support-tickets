import streamlit as st

# ✅ Move this line to be the first Streamlit command in the script!
st.set_page_config(page_title="AI-Powered Support Tickets", page_icon="🎫")

import datetime
import random
import altair as alt
import numpy as np
import pandas as pd
import openai  # ✅ Correct way to import OpenAI
import os
from dotenv import load_dotenv  # Import dotenv to load env variables
from sklearn.ensemble import RandomForestRegressor  # For time prediction
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# ✅ Load API Key Securely from Streamlit Secrets or .env for local
if "openai" in st.secrets and "api_key" in st.secrets["openai"]:
    api_key = st.secrets["openai"]["api_key"]
else:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

# ✅ Commented out API Key Debugging (No need to show after confirmation)
# if not api_key or "sk-" not in api_key:
#     st.error("❌ OpenAI API Key is missing or invalid! Ensure it's set in `.env` for local or Streamlit Secrets for deployment.")
# else:
#     st.success(f"✅ API Key Loaded: {api_key[:5]}...******")  # Debugging (only shows partial key)

# ✅ Initialize OpenAI client correctly
client = openai.Client(api_key=api_key)  # ✅ Corrected initialization

# Streamlit UI settings
st.title("🎫 AI-Powered Support Ticket System")

# Function to generate AI-powered solution recommendations
def get_ai_solutions(issue):
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an AI assistant for technical support."},
                {"role": "user", "content": f"Provide a possible solution for the issue: {issue}"},
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error fetching AI response: {str(e)}"

# Function to predict resolution time
# ✅ Function to train the resolution time model
def train_resolution_time_model(df):
    le_priority = LabelEncoder()
    le_status = LabelEncoder()

    # ✅ Ensure encoders know all possible values before fitting
    le_priority.fit(POSSIBLE_PRIORITIES)
    le_status.fit(POSSIBLE_STATUSES)  # Ensure "Open" is always in the classes

    df["Priority"] = df["Priority"].apply(lambda x: x if x in POSSIBLE_PRIORITIES else "Medium")
    df["Status"] = df["Status"].apply(lambda x: x if x in POSSIBLE_STATUSES else "Open")

    df["Priority"] = le_priority.transform(df["Priority"])
    df["Status"] = le_status.transform(df["Status"])

    return le_priority, le_status


# ✅ Limit Default Tickets to 5
if "df" not in st.session_state:
    np.random.seed(42)

    issue_descriptions = [
        "Network connectivity issues in the office",
        "Software application crashing on startup",
        "Printer not responding to print commands",
        "Email server downtime",
        "Data backup failure",
    ]

    data = {
        "ID": [f"TICKET-{i}" for i in range(1005, 1000, -1)],  # Only 5 tickets now
        "Issue": np.random.choice(issue_descriptions, size=5),
        "Status": np.random.choice(["Open", "In Progress", "Closed"], size=5),
        "Priority": np.random.choice(["High", "Medium", "Low"], size=5),
        "Date Submitted": [datetime.date(2023, 6, 1) + datetime.timedelta(days=random.randint(0, 30)) for _ in range(5)],
    }
    df = pd.DataFrame(data)
    st.session_state.df = df
    st.session_state.model, st.session_state.le_priority, st.session_state.le_status = train_resolution_time_model(df)

# ✅ Add a "Clear Tickets" Button
if st.button("🗑️ Clear All Tickets"):
    st.session_state.df = pd.DataFrame(columns=["ID", "Issue", "Status", "Priority", "Date Submitted"])
    st.success("✅ All tickets have been cleared.")

# Add a ticket form
st.header("Add a Ticket")

with st.form("add_ticket_form"):
    issue = st.text_area("Describe the issue")
    priority = st.selectbox("Priority", ["High", "Medium", "Low"])
    submitted = st.form_submit_button("Submit")

if submitted:
    recent_ticket_number = int(max(st.session_state.df.ID).split("-")[1])
    today = datetime.datetime.now().strftime("%m-%d-%Y")

    # Predict resolution time
    priority_encoded = st.session_state.le_priority.transform([priority])[0]
    status_encoded = st.session_state.le_status.transform(["Open"])[0]
    predicted_time = st.session_state.model.predict([[priority_encoded, status_encoded]])[0]

    # Get AI solution
    ai_solution = get_ai_solutions(issue)

    df_new = pd.DataFrame(
        [
            {
                "ID": f"TICKET-{recent_ticket_number+1}",
                "Issue": issue,
                "Status": "Open",
                "Priority": priority,
                "Date Submitted": today,
                "Predicted Resolution Time (hrs)": round(predicted_time, 2),
                "AI Suggested Solution": ai_solution,
            }
        ]
    )

    st.write("🎉 **Ticket submitted! Here are the details:**")
    st.dataframe(df_new, use_container_width=True, hide_index=True)

    st.session_state.df = pd.concat([df_new, st.session_state.df], axis=0)

# Show existing tickets
st.header("Existing Tickets")
st.write(f"Number of tickets: `{len(st.session_state.df)}`")

edited_df = st.data_editor(
    st.session_state.df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Status": st.column_config.SelectboxColumn("Status", options=["Open", "In Progress", "Closed"], required=True),
        "Priority": st.column_config.SelectboxColumn("Priority", options=["High", "Medium", "Low"], required=True),
    },
    disabled=["ID", "Date Submitted"],
)

# Show ticket stats
st.header("📊 Ticket Statistics")

col1, col2, col3 = st.columns(3)
num_open_tickets = len(st.session_state.df[st.session_state.df.Status == "Open"])
col1.metric(label="🟢 Open Tickets", value=num_open_tickets)
col2.metric(label="⏳ First Response Time (hrs)", value=5.2, delta=-1.5)
col3.metric(label="⏱️ Average Resolution Time (hrs)", value=16, delta=2)

# Show Altair charts
st.write("##### 📅 Ticket status per month")
status_plot = (
    alt.Chart(edited_df)
    .mark_bar()
    .encode(
        x="month(Date Submitted):O",
        y="count():Q",
        xOffset="Status:N",
        color="Status:N",
    )
    .configure_legend(orient="bottom", titleFontSize=14, labelFontSize=14, titlePadding=5)
)
st.altair_chart(status_plot, use_container_width=True, theme="streamlit")

st.write("##### 🔥 Current Ticket Priorities")
priority_plot = (
    alt.Chart(edited_df)
    .mark_arc()
    .encode(theta="count():Q", color="Priority:N")
    .properties(height=300)
    .configure_legend(orient="bottom", titleFontSize=14, labelFontSize=14, titlePadding=5)
)
st.altair_chart(priority_plot, use_container_width=True, theme="streamlit")
