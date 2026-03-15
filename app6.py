import streamlit as st
import pandas as pd
import numpy as np
import os
import base64
from PIL import Image
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

st.set_page_config(page_title="Smart Flight Finder", layout="wide")

# ======== BACKGROUND STYLING ========
def set_background(image_path="background.png"):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    st.markdown(f"""
        <style>
            .stApp {{
                background-image: url("data:image/jpg;base64,{encoded}");
                background-size: cover;
                background-attachment: fixed;
            }}
            .overlay {{
                background-color: rgba(255, 255, 255, 0.88);
                padding: 1.5rem;
                border-radius: 1rem;
                margin-bottom: 1rem;
                box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.1);
                backdrop-filter: blur(4px);
            }}
        </style>
    """, unsafe_allow_html=True)

set_background()

# ========== LOAD DATA ==========
@st.cache_data
def load_data():
    df = pd.read_csv("extended_flight_dataset.csv")
    df.columns = df.columns.str.strip()
    df = df[df["Source"] != df["Destination"]]
    df["Total_Duration_Minutes"] = df["Duration_hours"] * 60 + df["Duration_min"]
    return df

df = load_data()

# ========== MODELS ==========
@st.cache_resource
def train_model(data):
    X = data[["Airline", "Source", "Destination", "Total_Stops", "Duration_hours", "Duration_min"]].copy()
    y = data["Price"]
    encoders = {}
    for col in ["Airline", "Source", "Destination"]:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, encoders

model, label_encoders = train_model(df)

@st.cache_resource
def train_classifier(data):
    df = data.copy()
    kmeans = KMeans(n_clusters=3, random_state=42).fit(df[["Price", "Total_Duration_Minutes"]])
    df["Price_Cluster"] = kmeans.labels_
    means = df.groupby("Price_Cluster")["Price"].mean().sort_values()
    mapping = {means.index[0]: "Leisure", means.index[1]: "Mixed", means.index[2]: "Business"}
    df["Label"] = df["Price_Cluster"].map(mapping)

    # Balance classes
    max_per_class = 300
    df = df.groupby("Label").apply(lambda x: x.sample(n=min(len(x), max_per_class), random_state=42)).reset_index(drop=True)

    X = df[["Total_Stops", "Duration_hours", "Duration_min", "Price"]]
    y = df["Label"]
    clf = RandomForestClassifier(class_weight="balanced", random_state=42)
    clf.fit(X, y)
    return clf

classifier_model = train_classifier(df)

@st.cache_resource
def add_price_clusters(df):
    df = df.copy()
    cluster_data = df[["Price", "Total_Duration_Minutes"]]
    kmeans = KMeans(n_clusters=3, random_state=42).fit(cluster_data)
    df["Price_Cluster"] = kmeans.labels_
    means = df.groupby("Price_Cluster")["Price"].mean().sort_values()
    mapping = {means.index[0]: "💸 Budget", means.index[1]: "🎯 Standard", means.index[2]: "💎 Premium"}
    df["Price_Category"] = df["Price_Cluster"].map(mapping)
    return df

# ========== UTILITIES ==========
def display_logo(airline_name):
    logo_path = f"logos/{airline_name.lower().replace(' ', '_')}.png"
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode()
            return f"<img src='data:image/png;base64,{img_base64}' width='50' style='margin-right:10px;'/>"
    return ""

def render_flight(flight, clickable=False, key=None):
    logo_html = display_logo(flight["Airline"])
    html = f"""
    <div class='overlay' style='display: flex; align-items: center; justify-content: space-between; gap: 20px;'>
        <div style="display:flex;align-items:center;">{logo_html}</div>
        <div style="flex-grow:1;">
            <b>{flight['Airline']}</b><br>
            {flight['Source']} ➡️ {flight['Destination']}<br>
            🕒 {int(flight['Dep_hours']):02d}:{int(flight['Dep_min']):02d} → {int(flight['Arrival_hours']):02d}:{int(flight['Arrival_min']):02d}<br>
            ⏱️ {flight['Duration_hours']}h {flight['Duration_min']}m | Stops: {flight['Total_Stops']}<br>
            💰 Base: ₹{flight['Price']} | 🔮 Predicted: ₹{flight['Predicted Price (₹)']}<br>
            🏷️ Category: {flight.get("Price_Category", "❓")}<br>
            🧠 Suggested For: {flight.get("Traveler_Type", "❓")}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
    if clickable:
        return st.button("✈️ Select this flight", key=key)
    return False

def get_filtered_flights(source, destination, date):
    day = date.day
    month = date.month
    filtered = df[(df["Source"] == source) & (df["Destination"] == destination)
                  & (df["Date"] == day) & (df["Month"] == month)].copy()
    if filtered.empty:
        return filtered

    X = filtered[["Airline", "Source", "Destination", "Total_Stops", "Duration_hours", "Duration_min"]].copy()
    for col in ["Airline", "Source", "Destination"]:
        le = label_encoders[col]
        X[col] = X[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else 0)

    filtered["Predicted Price (₹)"] = model.predict(X).round().astype(int)
    filtered = add_price_clusters(filtered)

    X_class = filtered[["Total_Stops", "Duration_hours", "Duration_min", "Price"]]
    filtered["Traveler_Type"] = classifier_model.predict(X_class)
    return filtered

# ========== APP ==========
st.markdown("<h1 style='text-align:center;'>✈️ Smart Flight Finder</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Search. Compare. Discover smarter ways to fly.</p>", unsafe_allow_html=True)

# ===== STEP 1: Flight Search =====
with st.form("search_form"):
    col1, col2, col3 = st.columns(3)
    source = col1.selectbox("From", sorted(df["Source"].unique()))
    destination = col2.selectbox("To", sorted(df["Destination"].unique()))
    date = col3.date_input("Travel Date", value=datetime.today())
    submit = st.form_submit_button("🔍 Show Flights")

if submit:
    st.session_state["flights"] = get_filtered_flights(source, destination, date)
    st.session_state["selected_index"] = None
    st.session_state["selected_category"] = None
    st.session_state["time_slot"] = None

# ===== STEP 2: Show Flights =====
if "flights" in st.session_state and st.session_state.get("selected_index") is None:
    flights = st.session_state["flights"]
    if flights.empty:
        st.warning("No flights found.")
    else:
        st.markdown("### ✨ Recommended Flights")
        for i, (_, row) in enumerate(flights.head(10).iterrows()):
            if render_flight(row, clickable=True, key=f"flight_{i}"):
                st.session_state["selected_index"] = row.name

# ===== STEP 3: Choose Ticket Type =====
if st.session_state.get("selected_index") is not None and st.session_state.get("selected_category") is None:
    flight = st.session_state["flights"].loc[st.session_state["selected_index"]]
    base_price = flight["Predicted Price (₹)"]
    left, right = st.columns([2, 1])
    with left:
        render_flight(flight)

    with right:
        st.markdown("#### 🎫 Choose Ticket Type")
        col1, col2, col3 = st.columns(3)
        if col1.button("💸 Basic", key="basic_btn"):
            st.session_state["selected_category"] = "Basic"
        if col2.button("🎯 Standard", key="standard_btn"):
            st.session_state["selected_category"] = "Standard"
        if col3.button("💎 Premium", key="premium_btn"):
            st.session_state["selected_category"] = "Premium"
 

# ===== STEP 4: Time Slot Selection =====
if (
    st.session_state.get("selected_index") is not None
    and st.session_state.get("selected_category") is not None
    and st.session_state.get("time_slot") is None
):
    st.markdown(f"### 🕒 Choose Departure Time ({st.session_state['selected_category']})")
    col1, col2, col3, col4 = st.columns(4)
    if col1.button("🌅 Morning (6–12)", key="morning"):
        st.session_state["time_slot"] = "Morning"
    if col2.button("🌤️ Afternoon (12–18)", key="afternoon"):
        st.session_state["time_slot"] = "Afternoon"
    if col3.button("🌇 Evening (18–21)", key="evening"):
        st.session_state["time_slot"] = "Evening"
    if col4.button("🌙 Night (21–6)", key="night"):
        st.session_state["time_slot"] = "Night"

# ===== STEP 5: Final Results =====
if st.session_state.get("time_slot"):
    slot = st.session_state["time_slot"]
    selected_cat = st.session_state["selected_category"]
    filtered_final = st.session_state["flights"].copy()

    def is_slot(hour):
        if slot == "Morning":
            return 6 <= hour < 12
        elif slot == "Afternoon":
            return 12 <= hour < 18
        elif slot == "Evening":
            return 18 <= hour < 21
        elif slot == "Night":
            return hour >= 21 or hour < 6
        return False

    filtered_final = filtered_final[
        (filtered_final["Dep_hours"].apply(is_slot)) &
        (filtered_final["Price_Category"].str.contains(selected_cat, case=False))
    ]

    st.markdown(f"### ✅ {slot} {selected_cat} Flights")
    if filtered_final.empty:
        st.warning("No matching flights found.")
    else:
        for _, row in filtered_final.head(5).iterrows():
            render_flight(row)
