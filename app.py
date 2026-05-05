import streamlit as st
st.title("Real Estate Dashboard")
st.write("App is running ✅")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Jeddah Real Estate Dashboard", layout="wide")

# ---------------- TITLE ----------------
st.markdown("""
    <h1 style='text-align: center; color: #f4a6c1;'>
    🏡 Jeddah Real Estate Analytics Dashboard
    </h1>
""", unsafe_allow_html=True)

st.markdown("""
Welcome to an interactive real estate analytics dashboard exploring housing trends in Jeddah ✨
""")

# ---------------- DATA ----------------
np.random.seed(42)
n = 300

districts = ["Al Salamah", "Al Rawdah", "Al Hamra", "Al Shati", "Al Aziziyah", "Al Nuzhah", "Al Rehab"]

property_types = ["Apartment", "Villa"]

lat_map = {
    "Al Shati": 21.609,
    "Al Hamra": 21.543,
    "Al Rawdah": 21.575,
    "Al Salamah": 21.586,
    "Al Nuzhah": 21.616,
    "Al Aziziyah": 21.500,
    "Al Rehab": 21.470
}

lon_map = {
    "Al Shati": 39.110,
    "Al Hamra": 39.116,
    "Al Rawdah": 39.160,
    "Al Salamah": 39.150,
    "Al Nuzhah": 39.180,
    "Al Aziziyah": 39.200,
    "Al Rehab": 39.220
}

df = pd.DataFrame({
    "District": np.random.choice(districts, n),
    "Property_Type": np.random.choice(property_types, n),
    "Area_sqm": np.random.randint(70, 500, n),
})

base_price = {
    "Al Shati": 9000,
    "Al Hamra": 7500,
    "Al Rawdah": 6500,
    "Al Salamah": 6000,
    "Al Nuzhah": 5500,
    "Al Aziziyah": 4500,
    "Al Rehab": 5000
}

df["Price_per_sqm"] = df["District"].map(base_price) + np.random.randint(-500, 1500, n)
df["Price_SAR"] = df["Price_per_sqm"] * df["Area_sqm"]

df["lat"] = df["District"].map(lat_map)
df["lon"] = df["District"].map(lon_map)

# ---------------- SIDEBAR FILTERS ----------------
st.sidebar.header("💗 Filters")

district_filter = st.sidebar.multiselect(
    "Select District",
    options=df["District"].unique(),
    default=df["District"].unique()
)

type_filter = st.sidebar.multiselect(
    "Property Type",
    options=df["Property_Type"].unique(),
    default=df["Property_Type"].unique()
)

filtered_df = df[
    (df["District"].isin(district_filter)) &
    (df["Property_Type"].isin(type_filter))
]

# ---------------- KPI ----------------
st.markdown("### 💗 Key Insights")

col1, col2, col3 = st.columns(3)

col1.metric("Total Listings", len(filtered_df))
col2.metric("Avg Price (SAR)", int(filtered_df["Price_SAR"].mean()))
col3.metric("Avg Area (sqm)", int(filtered_df["Area_sqm"].mean()))

# ---------------- CHART 1 ----------------
st.markdown("### 📊 Average Price by District")

fig, ax = plt.subplots(figsize=(10,5))
filtered_df.groupby("District")["Price_SAR"].mean().sort_values().plot(kind="bar", ax=ax, color="#f4a6c1")
plt.xticks(rotation=45)
st.pyplot(fig)

# ---------------- CHART 2 ----------------
st.markdown("### 📈 Area vs Price")

fig2, ax2 = plt.subplots()
sns.scatterplot(
    data=filtered_df,
    x="Area_sqm",
    y="Price_SAR",
    hue="Property_Type",
    ax=ax2
)
st.pyplot(fig2)

# ---------------- CHART 3 ----------------
st.markdown("### 🏠 Property Type Distribution")

fig3, ax3 = plt.subplots()
filtered_df["Property_Type"].value_counts().plot(
    kind="pie",
    autopct='%1.1f%%',
    ax=ax3,
    colors=["#f4a6c1", "#ffd6e7"]
)
ax3.set_ylabel("")
st.pyplot(fig3)

# ---------------- MAP ----------------
st.markdown("### 🗺️ Jeddah Map (District Locations)")

st.map(filtered_df[["lat", "lon"]])

# ---------------- ML MODEL ----------------
st.markdown("### 💰 Predict Property Price")

model_df = df.copy()

model_df["Property_Type"] = model_df["Property_Type"].map({"Apartment": 0, "Villa": 1})

X = model_df[["Area_sqm", "Property_Type"]]
y = model_df["Price_SAR"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

area_input = st.slider("Select Area (sqm)", 70, 500, 120)
type_input = st.selectbox("Property Type", ["Apartment", "Villa"])

type_val = 0 if type_input == "Apartment" else 1

predicted_price = model.predict([[area_input, type_val]])

st.success(f"💰 Predicted Price: {int(predicted_price[0]):,} SAR")

# ---------------- DOWNLOAD BUTTON ----------------
st.markdown("### 📥 Download Data")

csv = filtered_df.to_csv(index=False).encode('utf-8')

st.download_button(
    label="Download Dataset",
    data=csv,
    file_name="jeddah_real_estate_data.csv",
    mime="text/csv"
)

# ---------------- TABLE ----------------
st.markdown("### 📋 Data Table")

st.markdown("### 📋 Data Table")

st.write("Number of rows:", len(filtered_df))

if filtered_df.empty:
    st.warning("No data found. The filters may be hiding all rows.")
else:
    st.dataframe(filtered_df, use_container_width=True)