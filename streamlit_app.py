import streamlit as st
import datetime
import random
import altair as alt
import numpy as np
import pandas as pd
import openai
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ✅ Set Streamlit UI settings
st.set_page_config(page_title="AI-Powered Support Tickets", page_icon="🎫", layout="wide")

# ✅ Load API Key from GitHub Codespaces Secrets
api_key = os.getenv("OPENAI_API_KEY")

# ✅ If API key is missing, show a warning and disable AI features
if not api_key:
    st.warning("⚠️ OpenAI API key is missing! Set `OPENAI_API_KEY` as a GitHub Codespaces Secret.")
else:
    openai.api_key = api_key
    st.success("✅ OpenAI API key loaded successfully!")

# ✅ Define possible labels
POSSIBLE_PRIORITIES = ["High", "Medium", "Low"]
POSSIBLE_STATUSES = ["Open", "In Progress", "Closed"]

# ✅ Function to train the resolution time model
def train_resolution_time_model(df):
    le_priority = LabelEncoder()
    le_status = LabelEncoder()
    
    le_priority.fit(POSSIBLE_PRIORITIES)
    le_status.fit(POSSIBLE_STATUSES)

    df["Priority"] = df["Priority"].apply(lambda x: x if x in POSSIBLE_PRIORITIES else "Medium")
    df["Status"] = df["Status"].apply(lambda x: x if x in POSSIBLE_STATUSES else "Open")

    df["Priority"] = le_priority.transform(df["Priority"])
    df["Status"] = le_status.transform(df["Status"])

    X = df[["Priority", "Status"]]
    y = np.random.randint(2, 48, size=len(df))  # Simulated resolution time

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, le_priority, le_status

# ✅ Function to generate AI-powered solutions
def get_ai_solutions(issue):
    if not api_key:
        return "⚠️ AI support is disabled due to missing API key."

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an AI assistant for technical support."},
                {"role": "user", "content": f"Provide a possible solution for the issue: {issue}"},
            ]
        )
        return response.choices[0].message["content"]
    except Exception as e:
        return f"❌ Error fetching AI response: {str(e)}"

# ✅ Generate initial dataset
issue_descriptions = [
    "Network connectivity issues in the office",
    "Software application crashing on startup",
    "Printer not responding to print commands",
    "Email server downtime",
    "Data backup failure",
]

data = {
    "ID": [f"TICKET-{i}" for i in range(1005, 1000, -1)],
    "Issue": np.random.choice(issue_descriptions, size=5),
    "Status": np.random.choice(POSSIBLE_STATUSES, size=5),
    "Priority": np.random.choice(POSSIBLE_PRIORITIES, size=5),
    "Date Submitted": [datetime.date(2023, 6, 1) + datetime.timedelta(days=random.randint(0, 30)) for _ in range(5)],
}

df = pd.DataFrame(data)
model, le_priority, le_status = train_resolution_time_model(df)

# ✅ "Clear All Tickets" Button
if st.button("🗑️ Clear All Tickets"):
    df = pd.DataFrame(columns=["ID", "Issue", "Status", "Priority", "Date Submitted"])
    st.success("✅ All tickets have been cleared.")

# ✅ Add a Ticket Form
st.header("Add a Ticket")

with st.form("add_ticket_form"):
    issue = st.text_area("Describe the issue")
    priority = st.selectbox("Priority", POSSIBLE_PRIORITIES)
    submitted = st.form_submit_button("Submit")

if submitted:
    recent_ticket_number = int(max(df.ID).split("-")[1]) if len(df) > 0 else 1006
    today = datetime.datetime.now().strftime("%m-%d-%Y")

    if priority not in le_priority.classes_:
        st.warning(f"Unknown priority '{priority}', defaulting to 'Medium'.")
        priority = "Medium"
    priority_encoded = le_priority.transform([priority])[0]

    if "Open" not in le_status.classes_:
        st.warning("Status 'Open' is missing in encoder, refitting model.")
        le_status.fit(POSSIBLE_STATUSES)
    status_encoded = le_status.transform(["Open"])[0]

    predicted_time = model.predict([[priority_encoded, status_encoded]])[0]
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

    df = pd.concat([df_new, df], axis=0)

# ✅ Show Existing Tickets
st.header("Existing Tickets")
st.write(f"Number of tickets: `{len(df)}`")

edited_df = st.data_editor(
    df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Status": st.column_config.SelectboxColumn("Status", options=POSSIBLE_STATUSES, required=True),
        "Priority": st.column_config.SelectboxColumn("Priority", options=POSSIBLE_PRIORITIES, required=True),
    },
    disabled=["ID", "Date Submitted"],
)

# ✅ Show Ticket Stats
st.header("📊 Ticket Statistics")

col1, col2, col3 = st.columns(3)
num_open_tickets = len(df[df.Status == "Open"])
col1.metric(label="🟢 Open Tickets", value=num_open_tickets)
col2.metric(label="⏳ First Response Time (hrs)", value=5.2, delta=-1.5)
col3.metric(label="⏱️ Average Resolution Time (hrs)", value=16, delta=2)
