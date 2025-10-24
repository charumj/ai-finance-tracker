# streamlit_finmind.py
import streamlit as st
import pandas as pd
import os
import csv
import json
import re
from datetime import datetime
import matplotlib.pyplot as plt

# ---------- Optional AI ----------
AI_AVAILABLE = False
classifier = None
USE_AI_DEFAULT = False  # default OFF to avoid long downloads during demo

# Try to import transformers lazily when user enables AI in UI
def try_load_transformer():
    global AI_AVAILABLE, classifier
    try:
        from transformers import pipeline
        with st.spinner("Loading transformer model (first time may take a while)..."):
            classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        AI_AVAILABLE = True
    except Exception as e:
        AI_AVAILABLE = False
        st.error("Transformer load failed — falling back to keyword categorizer. (This is fine for demo.)")
        st.write(f"Debug: {e}")

# ---------- Data directory & helpers ----------
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
    # find first number
    m = re.search(r'([0-9]+(?:\.[0-9]+)?)', s)
    if not m:
        return None
    num = float(m.group(1))
    # check for unit around number
    after = s[m.end():]
    before = s[:m.start()]
    for token in UNIT_MAP:
        if token in after or token in before:
            return num * UNIT_MAP[token]
    # suffix check (e.g., 5k, 3l)
    m2 = re.search(r'([0-9]+(?:\.[0-9]+)?)(k|m|cr|l|lac|lakh)', s)
    if m2:
        suf = m2.group(2)
        if suf == 'k':
            return num * 1000
        if suf in ['l', 'lac', 'lakh']:
            return num * 100000
        if suf == 'm':
            return num * 1000000
        if suf == 'cr':
            return num * 10000000
    return num

# ---------- Extract items from free text ----------
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
        found = False
        for pat in EXTRACT_PATTERNS:
            m = re.search(pat, p)
            if m:
                g1 = m.group(1).strip()
                g2 = m.group(2).strip()
                # assume g2 is amount
                amt = parse_amount_text(g2)
                desc = g1
                if amt is None:
                    alt = parse_amount_text(g1)
                    if alt is not None:
                        amt = alt
                        desc = g2
                if amt is not None:
                    items.append((desc.strip(), float(amt)))
                    found = True
                    break
        if not found:
            # fallback: find number and rest as desc
            mnum = re.search(r'([0-9,]+(?:\.[0-9]+)?)', p)
            if mnum:
                amt = parse_amount_text(mnum.group(1))
                desc = p.replace(mnum.group(1), "").strip(" -:₹rs.")
                if amt is not None and desc:
                    items.append((desc.strip(), float(amt)))
    return items

# ---------- Categorization ----------
DEFAULT_CATEGORIES = ["Food", "Shopping", "Transport", "Housing", "Utilities", "Entertainment", "Education", "Other"]

def keyword_categorizer(text):
    text = text.lower()
    if any(w in text for w in ["food","pizza","burger","restaurant","dinner","lunch","zomato","swiggy","grocery","milk","egg"]):
        return "Food"
    if any(w in text for w in ["biba","zara","flipkart","amazon","shirt","jeans","dress","shopping","mall"]):
        return "Shopping"
    if any(w in text for w in ["uber","ola","bus","train","flight","taxi","auto","metro","travel"]):
        return "Transport"
    if any(w in text for w in ["rent","emi","home","house"]):
        return "Housing"
    if any(w in text for w in ["internet","recharge","mobile","wifi","electricity","bill","utility"]):
        return "Utilities"
    if any(w in text for w in ["movie","netflix","spotify","game","concert"]):
        return "Entertainment"
    if any(w in text for w in ["book","course","tuition","study","exam"]):
        return "Education"
    return "Other"

def categorize_text(text, use_ai=False):
    if use_ai and AI_AVAILABLE and classifier is not None:
        try:
            res = classifier(text, DEFAULT_CATEGORIES)
            return res["labels"][0]
        except Exception:
            return keyword_categorizer(text)
    else:
        return keyword_categorizer(text)

# ---------- Data operations ----------
def append_expense(user, date, desc, amount, category):
    ensure_user_files(user)
    ef = expenses_file_for(user)
    with open(ef, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([date, desc.title(), amount, category])

def import_csv_for_user(user, path):
    df = pd.read_csv(path)
    # map columns
    lower_map = {c.lower(): c for c in df.columns}
    desc_col = amt_col = None
    for k in lower_map:
        if 'desc' in k:
            desc_col = lower_map[k]
        if 'amount' in k or 'amt' in k:
            amt_col = lower_map[k]
    if desc_col is None or amt_col is None:
        st.error("CSV must contain columns with 'Description' and 'Amount' (or similar).")
        return
    for _, row in df.iterrows():
        desc = str(row[desc_col])
        raw_amt = row[amt_col]
        try:
            amt = float(str(raw_amt).replace(",", "").strip())
        except:
            amt = parse_amount_text(str(raw_amt))
            if amt is None:
                continue
        cat = categorize_text(desc, use_ai=st.session_state.get("use_ai", USE_AI_DEFAULT))
        append_expense(user, datetime.now().strftime("%Y-%m-%d"), desc.title(), amt, cat)
    st.success("CSV imported and categorized.")

# ---------- Goal operations ----------
def set_goal_for_user(user, goal_name, target_amount, monthly_income):
    ensure_user_files(user)
    gf = goal_file_for(user)
    data = {}
    if os.path.exists(gf) and os.path.getsize(gf) > 0:
        with open(gf, "r") as f:
            data = json.load(f)
    data['goal_name'] = goal_name
    data['target_amount'] = float(target_amount)
    data['monthly_income'] = float(monthly_income)
    data['progress'] = data.get('progress', 0.0)
    data['last_updated_month'] = data.get('last_updated_month', "")
    with open(gf, "w") as f:
        json.dump(data, f, indent=2)
    st.success(f"Goal set: {goal_name} — ₹{float(target_amount):.2f}")

# ---------- Analysis ----------
def read_user_expenses_df(user):
    ef = expenses_file_for(user)
    if not os.path.exists(ef):
        return pd.DataFrame(columns=["Date", "Description", "Amount", "Category"])
    df = pd.read_csv(ef, parse_dates=["Date"])
    return df

def analyze_user(user):
    df = read_user_expenses_df(user)
    if df.empty:
        st.info("No expenses yet.")
        return
    # ensure Date dtype
    if df['Date'].dtype == object:
        try:
            df['Date'] = pd.to_datetime(df['Date'])
        except:
            pass

    today = pd.Timestamp(datetime.now().date())
    last_30 = df[df['Date'] >= (today - pd.Timedelta(days=30))]

    # pie chart
    pie_data = last_30.groupby("Category")["Amount"].sum()
    if pie_data.sum() > 0:
        fig1, ax1 = plt.subplots()
        pie_data.plot(kind="pie", autopct='%1.1f%%', ax=ax1)
        ax1.set_ylabel("")
        ax1.set_title("Spending by Category (Last 30 days)")
        st.pyplot(fig1)
    else:
        st.write("No spending in last 30 days to show pie chart.")

    # monthly bar
    df['Month'] = pd.to_datetime(df['Date']).dt.to_period('M')
    monthly = df.groupby('Month')['Amount'].sum().sort_index()
    if len(monthly) > 0:
        fig2, ax2 = plt.subplots(figsize=(8,3))
        monthly.index = monthly.index.astype(str)
        monthly.plot(kind='bar', ax=ax2)
        ax2.set_title("Monthly Spending (All time)")
        ax2.set_ylabel("Amount (₹)")
        ax2.set_xlabel("Month")
        plt.xticks(rotation=45)
        st.pyplot(fig2)

    st.write("### Top categories (last 30 days)")
    st.write(last_30.groupby("Category")["Amount"].sum().sort_values(ascending=False).head(10))

    # update goal progress
    gf = goal_file_for(user)
    if os.path.exists(gf) and os.path.getsize(gf) > 0:
        with open(gf, "r") as f:
            goal = json.load(f)
    else:
        goal = {}

    if goal and goal.get('target_amount'):
        cur_month = today.strftime("%Y-%m")
        last_updated = goal.get('last_updated_month', "")
        # compute expenses in current month
        month_start = today.replace(day=1)
        this_month_exp = df[df['Date'] >= month_start]['Amount'].sum()
        monthly_income = float(goal.get('monthly_income', 0))
        # compute savings this month
        savings_this_month = max(0.0, monthly_income - this_month_exp)
        if last_updated != cur_month:
            goal['progress'] = float(goal.get('progress', 0.0)) + savings_this_month
            goal['last_updated_month'] = cur_month
            with open(gf, "w") as f:
                json.dump(goal, f, indent=2)
            st.success(f"Monthly progress updated: approx saved ₹{savings_this_month:.2f} added to your goal.")
        else:
            st.info("Monthly progress already recorded for this month.")

        remaining = float(goal['target_amount']) - float(goal.get('progress', 0.0))
        st.write(f"**Goal:** {goal.get('goal_name')} — Target ₹{goal.get('target_amount'):.2f}")
        st.write(f"**Progress:** ₹{goal.get('progress', 0.0):.2f}  |  **Remaining:** ₹{remaining:.2f}")
        # estimate months left
        expected_monthly_saving = max(1e-9, monthly_income - this_month_exp)
        est_months_left = remaining / expected_monthly_saving if expected_monthly_saving > 0 else float('inf')
        st.write(f"Estimated months to reach goal (at current saving rate): {int(est_months_left) if est_months_left!=float('inf') else 'Unreachable at current rate'}")

        # recommendation
        cat_spend = last_30.groupby("Category")["Amount"].sum().sort_values(ascending=False)
        if not cat_spend.empty:
            top_cat = cat_spend.index[0]
            top_amt = cat_spend.iloc[0]
            reduce_pct = 0.15
            monthly_saving_if_reduced = top_amt * reduce_pct
            st.write(f"**Tip:** You spent ₹{top_amt:.2f} on {top_cat} in the last 30 days.")
            st.write(f"If you reduce {top_cat} by {int(reduce_pct*100)}%, you may save approx ₹{monthly_saving_if_reduced:.2f} monthly and reach your goal faster.")
    else:
        st.info("No goal set. Use Set Goal to start tracking.")

# ---------- Auth UI ----------
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
            st.error("Username exists. Please login or choose another.")
            return
        pd.DataFrame([[new_user, new_pass]], columns=["username", "password"]).to_csv(USERS_FILE, mode="a", header=False, index=False)
        st.success("Signup complete. Please login in the sidebar.")

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

# ---------- Main app ----------
st.set_page_config(page_title="FinMind - AI Personal Finance", layout="wide")
st.title("FinMind — AI Personal Finance Companion")

# sidebar: login / options
st.sidebar.header("Account")
if 'user' not in st.session_state:
    # show signup + login
    login_ui()
    st.sidebar.markdown("---")
    st.sidebar.write("New user?")
    signup_ui()
else:
    st.sidebar.markdown(f"**Logged in as:** {st.session_state['user']}")
    if st.sidebar.button("Logout (main)"):
        st.session_state.pop('user', None)
        st.experimental_rerun()

# AI toggle
st.sidebar.markdown("---")
use_ai = st.sidebar.checkbox("Use AI categorization (Transformer)", value=USE_AI_DEFAULT)
st.session_state["use_ai"] = use_ai
if use_ai and not AI_AVAILABLE:
    try_load_transformer()

# If not logged in, show intro
if 'user' not in st.session_state:
    st.info("Please sign up or login using the controls in the sidebar. Once logged in you can add expenses, upload CSVs, set goals and analyze.")
    st.markdown("**Quick demo steps:** 1) Signup 2) Login 3) Click 'Add expenses' below")
    st.stop()

# Logged-in UI: tabs
user = st.session_state['user']
ensure_user_files(user)

tab = st.radio("Choose action", ["Add expenses", "Upload CSV", "Set goal", "Analyze", "Show recent"])

if tab == "Add expenses":
    st.subheader("Add expenses in one line")
    st.write("Example formats: `bought dress for 4000, milk 200, rent 5000` or `pizza ₹350, uber 120`")
    text = st.text_area("Enter expenses", height=120)
    if st.button("Save expenses"):
        if not text.strip():
            st.error("Enter an expense line first.")
        else:
            items = extract_items_from_text(text)
            if not items:
                st.warning("No items parsed — try format 'item for amount' or 'item amount'.")
            else:
                for desc, amt in items:
                    cat = categorize_text(desc, use_ai=st.session_state.get("use_ai", USE_AI_DEFAULT))
                    append_expense(user, datetime.now().strftime("%Y-%m-%d"), desc.title(), amt, cat)
                    st.success(f"Saved: {desc.title()} | ₹{amt:.2f} | {cat}")
                st.balloons()

elif tab == "Upload CSV":
    st.subheader("Upload CSV of expenses")
    st.write("CSV should contain Description and Amount columns (column names can vary).")
    uploaded_file = st.file_uploader("Choose CSV file", type="csv")
    if uploaded_file is not None:
        # save to temp then import
        tmp_path = os.path.join(DATA_DIR, f"tmp_upload_{user}.csv")
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        import_csv_for_user(user, tmp_path)
        os.remove(tmp_path)

elif tab == "Set goal":
    st.subheader("Set / update financial goal")
    with st.form("goal_form"):
        gname = st.text_input("Goal name (e.g., Buy a car)")
        gtarget = st.text_input("Target amount (e.g., 300000 or '3 lakh')")
        ginc = st.text_input("Monthly income (e.g., 30000)")
        submitted = st.form_submit_button("Set goal")
        if submitted:
            targ = parse_amount_text(gtarget)
            inc = parse_amount_text(ginc)
            if targ is None:
                st.error("Could not parse target amount. Use numbers like 300000 or '3 lakh'.")
            elif inc is None:
                st.error("Could not parse income. Use numbers like 30000.")
            else:
                set_goal_for_user(user, gname, targ, inc)

elif tab == "Analyze":
    st.subheader("Analyze spending & goal progress")
    analyze_user(user)

elif tab == "Show recent":
    st.subheader("Recent expenses")
    df = read_user_expenses_df(user)
    if df.empty:
        st.info("No expenses yet.")
    else:
        st.dataframe(df.sort_values(by="Date", ascending=False).head(50))
