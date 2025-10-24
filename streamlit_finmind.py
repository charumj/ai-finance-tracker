import streamlit as st
import pandas as pd
import os
import csv
import json
from datetime import datetime
import re
import plotly.express as px
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# ---------- Data directory ----------
DATA_DIR = "finmind_streamlit_data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

USERS_FILE = os.path.join(DATA_DIR, "users.csv")

def ensure_users_file():
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["username", "password"])
ensure_users_file()

# ---------- File helpers ----------
def expenses_file_for(user):
    return os.path.join(DATA_DIR, f"expenses_{user}.csv")

def goal_file_for(user):
    return os.path.join(DATA_DIR, f"goal_{user}.json")

def ensure_user_files(user):
    ef = expenses_file_for(user)
    if not os.path.exists(ef):
        with open(ef, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Date", "Description", "Amount", "Category"])
    gf = goal_file_for(user)
    if not os.path.exists(gf):
        with open(gf, "w") as f:
            json.dump({}, f)

# ---------- Amount parsing ----------
UNIT_MAP = {
    "k": 1000, "thousand": 1000,
    "lakh": 100000, "lakhs": 100000, "lac": 100000,
    "crore": 10000000, "cr": 10000000
}

def parse_amount_text(s: str):
    s = str(s).replace(",", "").lower().strip()
    try:
        return float(s)
    except:
        pass
    for token, mult in UNIT_MAP.items():
        if token in s:
            num = ''.join([c for c in s if c.isdigit() or c == '.'])
            if num:
                return float(num) * mult
    return None

# ---------- AI Categorization ----------
MODEL_DIR = os.path.join(os.getcwd(), "expense_model")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
except Exception as e:
    st.error("⚠️ Failed to load model. Ensure 'expense_model' folder exists with model files.")
    st.stop()

def categorize_text(text):
    res = classifier(text)
    return res[0]['label']

# ---------- Extract items ----------
EXTRACT_PATTERNS = [
    r'([\w\s&.-]+?)\s*(?:for|:|-)\s*₹?\s*([0-9,]+(?:\.[0-9]+)?)',
    r'([\w\s&.-]+?)\s*₹\s*([0-9,]+(?:\.[0-9]+)?)',
    r'([0-9,]+(?:\.[0-9]+)?)\s*(?:rs|rupees|₹)\s*for\s*([\w\s&.-]+?)',
    r'([\w\s&.-]+?)\s+([0-9,]+(?:\.[0-9]+)?)'
]

def extract_items_from_text(text):
    text = text.lower()
    items = []
    parts = re.split(r',|\band\b', text)
    for part in parts:
        p = part.strip()
        for pat in EXTRACT_PATTERNS:
            m = re.search(pat, p)
            if m:
                desc, amt_text = m.group(1).strip(), m.group(2).strip()
                amt = parse_amount_text(amt_text)
                if amt is not None:
                    items.append((desc.title(), float(amt)))
                break
    return items

# ---------- Data operations ----------
def append_expense(user, date, desc, amount, category):
    ensure_user_files(user)
    ef = expenses_file_for(user)
    with open(ef, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([date, desc, amount, category])

def import_csv_for_user(user, path):
    df = pd.read_csv(path)
    desc_col = next((c for c in df.columns if 'desc' in c.lower()), None)
    amt_col = next((c for c in df.columns if 'amount' in c.lower() or 'amt' in c.lower()), None)
    if not desc_col or not amt_col:
        st.error("CSV must contain Description and Amount columns.")
        return
    for _, row in df.iterrows():
        desc = str(row[desc_col])
        amt = parse_amount_text(str(row[amt_col]))
        if amt is None:
            continue
        cat = categorize_text(desc)
        append_expense(user, datetime.now().strftime("%Y-%m-%d"), desc, amt, cat)
    st.success("✅ CSV imported and categorized successfully!")

# ---------- Goal operations ----------
def set_goal_for_user(user, goal_name, target_amount, monthly_income):
    ensure_user_files(user)
    gf = goal_file_for(user)
    data = {'goal_name': goal_name, 'target_amount': float(target_amount), 'monthly_income': float(monthly_income)}
    with open(gf, "w") as f:
        json.dump(data, f, indent=2)
    st.success(f"🎯 Goal set: {goal_name} — ₹{float(target_amount):,.0f}")

# ---------- Read Data ----------
def read_user_expenses_df(user):
    ef = expenses_file_for(user)
    if not os.path.exists(ef):
        return pd.DataFrame(columns=["Date", "Description", "Amount", "Category"])
    return pd.read_csv(ef, parse_dates=["Date"])

# ---------- Analysis ----------
def analyze_user(user):
    df = read_user_expenses_df(user)
    if df.empty:
        st.info("No expenses yet. Add some to see analytics 💡")
        return

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Month"] = df["Date"].dt.to_period("M").astype(str)
    last_30 = df[df["Date"] >= (datetime.now() - pd.Timedelta(days=30))]

    # Load goal
    goal_path = goal_file_for(user)
    goal_name, goal_amount, monthly_income = "Your Goal", 0, 0
    if os.path.exists(goal_path) and os.path.getsize(goal_path) > 0:
        with open(goal_path, "r") as f:
            data = json.load(f)
        goal_name = data.get("goal_name", "Your Goal")
        goal_amount = data.get("target_amount", 0)
        monthly_income = data.get("monthly_income", 0)

    st.markdown("## 💰 Monthly Overview")
    total_spent = df["Amount"].sum()
    monthly_saving = max(monthly_income - total_spent, 0)
    progress = min((monthly_saving / goal_amount) * 100, 100) if goal_amount > 0 else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Income", f"₹{monthly_income:,.0f}")
    col2.metric("Spent", f"₹{total_spent:,.0f}")
    col3.metric("Saved", f"₹{monthly_saving:,.0f}")

    st.progress(progress / 100)
    st.markdown(f"🎯 **{goal_name}** — Goal ₹{goal_amount:,.0f} | Progress **{progress:.1f}%**")

    # Spending visualization
    if not last_30.empty:
        cat_spend = last_30.groupby("Category")["Amount"].sum().sort_values(ascending=False)
        if not cat_spend.empty:
            fig = px.pie(values=cat_spend.values, names=cat_spend.index,
                         title="Spending by Category (Last 30 days)", hole=0.4,
                         color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig, use_container_width=True)

    # Recommendations
    if not last_30.empty and monthly_income > 0:
        st.markdown("### 🧭 Smart Recommendations")
        cat_spend = last_30.groupby("Category")["Amount"].sum()
        total_expense = cat_spend.sum()
        for cat, amt in cat_spend.items():
            pct = (amt / total_expense) * 100
            if pct > 25:
                st.warning(f"⚠️ {pct:.1f}% spent on **{cat}** — reduce by 20% to save ₹{amt*0.2:,.0f}/month.")
            elif pct > 10:
                st.info(f"💡 {cat} expenses moderate ({pct:.1f}%). Reducing 10% saves ₹{amt*0.1:,.0f}/month.")
            else:
                st.success(f"✅ {cat} ({pct:.1f}%) — well managed.")

    # Forecast
    st.markdown("### 📅 Goal Forecast")
    if monthly_income == 0:
        st.warning("Please set your income to calculate forecast.")
    elif monthly_saving <= 0:
        st.error("Expenses exceed income. Reduce major categories.")
    else:
        months = (goal_amount - monthly_saving) / monthly_saving if goal_amount > 0 else 0
        st.success(f"📈 You’ll reach your goal in ~**{months:.1f} months** at current savings rate.")

# ---------- Auth ----------
def signup_ui():
    st.subheader("Create account")
    new_user = st.text_input("Username", key="su_user")
    new_pass = st.text_input("Password", type="password", key="su_pass")
    if st.button("Sign up"):
        df = pd.read_csv(USERS_FILE)
        if new_user.strip() == "":
            st.error("Enter a username")
            return
        if new_user in df['username'].values:
            st.error("Username exists.")
            return
        pd.DataFrame([[new_user, new_pass]], columns=["username", "password"]).to_csv(USERS_FILE, mode="a", header=False, index=False)
        st.success("Signup complete. Please login.")

def login_ui():
    st.sidebar.subheader("Login")
    user = st.sidebar.text_input("Username", key="login_user")
    pw = st.sidebar.text_input("Password", type="password", key="login_pw")
    if st.sidebar.button("Login"):
        df = pd.read_csv(USERS_FILE)
        if ((df['username'] == user) & (df['password'] == pw)).any():
            st.session_state['user'] = user
            ensure_user_files(user)
            st.sidebar.success(f"Logged in as {user}")
        else:
            st.sidebar.error("Invalid credentials")
    if st.sidebar.button("Logout"):
        st.session_state.pop('user', None)
        st.sidebar.info("Logged out")

# ---------- Streamlit main ----------
st.set_page_config(page_title="FinMind - AI Expense Tracker", layout="wide")
st.title("💸 FinMind — AI-Powered Expense Tracker")

# Sidebar login
st.sidebar.header("Account")
if 'user' not in st.session_state:
    login_ui()
    st.sidebar.markdown("---")
    st.sidebar.write("New user?")
    signup_ui()
else:
    st.sidebar.markdown(f"**Logged in as:** {st.session_state['user']}")
    if st.sidebar.button("Logout (main)"):
        st.session_state.pop('user', None)
        st.experimental_rerun()

if 'user' not in st.session_state:
    st.info("Please sign up or login first.")
    st.stop()

user = st.session_state['user']
ensure_user_files(user)

tab = st.radio("Choose Action", ["Add expenses", "Upload CSV", "Set goal", "Analyze", "Show recent"])

if tab == "Add expenses":
    st.subheader("Add expenses in one line")
    st.write("Example: `bought dress for 4000, milk 200, rent 5000`")
    text = st.text_area("Enter expenses", height=120)
    if st.button("Save expenses"):
        if text.strip():
            items = extract_items_from_text(text)
            if not items:
                st.warning("No valid items found — use format 'item for amount'.")
            else:
                for desc, amt in items:
                    cat = categorize_text(desc)
                    append_expense(user, datetime.now().strftime("%Y-%m-%d"), desc, amt, cat)
                    st.success(f"Saved: {desc} | ₹{amt:.2f} | {cat}")

elif tab == "Upload CSV":
    st.subheader("Upload CSV of expenses")
    uploaded_file = st.file_uploader("Choose CSV file", type="csv")
    if uploaded_file is not None:
        tmp_path = os.path.join(DATA_DIR, f"tmp_upload_{user}.csv")
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        import_csv_for_user(user, tmp_path)
        os.remove(tmp_path)

elif tab == "Set goal":
    st.subheader("Set / update financial goal")
    with st.form("goal_form"):
        gname = st.text_input("Goal name (e.g., Buy a car)")
        gtarget = st.text_input("Target amount (e.g., 3 lakh)")
        ginc = st.text_input("Monthly income (e.g., 30000)")
        submitted = st.form_submit_button("Set goal")
        if submitted:
            targ = parse_amount_text(gtarget)
            inc = parse_amount_text(ginc)
            if targ is None or inc is None:
                st.error("Could not parse amounts.")
            else:
                set_goal_for_user(user, gname, targ, inc)

elif tab == "Analyze":
    st.subheader("Analyze spending")
    analyze_user(user)

elif tab == "Show recent":
    st.subheader("Recent expenses")
    df = read_user_expenses_df(user)
    if df.empty:
        st.info("No expenses yet.")
    else:
        st.dataframe(df.sort_values(by="Date", ascending=False).head(50))
