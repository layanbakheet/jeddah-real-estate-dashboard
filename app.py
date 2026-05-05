import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Saudi Real Estate Analytics Dashboard", layout="wide")

# ---------------- CUSTOM STYLE ----------------
st.markdown("""
<style>

/* Main page background */
.stApp {
    background-color: #F7F5EF;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #102A43;
}

/* Sidebar text */
section[data-testid="stSidebar"] * {
    color: #F7F5EF;
}

/* Main title */
.main-title {
    text-align: center;
    color: #102A43;
    font-size: 44px;
    font-weight: 800;
    margin-bottom: 6px;
    letter-spacing: 0.5px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #52616B;
    font-size: 18px;
    margin-bottom: 32px;
}

/* Section headers */
h3 {
    color: #102A43;
    font-weight: 750;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #D8C690;
    padding: 18px;
    border-radius: 16px;
    box-shadow: 0px 6px 18px rgba(16, 42, 67, 0.10);
}

/* Metric label */
div[data-testid="stMetricLabel"] {
    color: #52616B;
    font-weight: 600;
}

/* Metric value */
div[data-testid="stMetricValue"] {
    color: #102A43;
    font-weight: 850;
}

/* Download button */
.stDownloadButton button {
    background-color: #102A43;
    color: #FFFFFF;
    border-radius: 10px;
    border: 1px solid #C9A646;
    padding: 10px 18px;
    font-weight: 650;
}

.stDownloadButton button:hover {
    background-color: #0B1F33;
    color: #C9A646;
    border: 1px solid #C9A646;
}

/* Dataframe */
div[data-testid="stDataFrame"] {
    border-radius: 12px;
    border: 1px solid #D8C690;
}

/* Select boxes */
div[data-baseweb="select"] {
    border-radius: 10px;
}

/* Slider color */
.stSlider [data-baseweb="slider"] {
    color: #C9A646;
}

/* Hide extra Streamlit style noise */
header[data-testid="stHeader"] {
    background-color: #F7F5EF;
}
/* Insight cards */
.insight-card {
    background-color: #FFFFFF;
    border: 1px solid #D8C690;
    border-left: 5px solid #C9A646;
    padding: 18px;
    border-radius: 16px;
    box-shadow: 0px 6px 18px rgba(16, 42, 67, 0.08);
    min-height: 145px;
}

.insight-title {
    color: #52616B;
    font-size: 14px;
    font-weight: 650;
    margin-bottom: 8px;
}

.insight-value {
    color: #102A43;
    font-size: 22px;
    font-weight: 850;
    margin-bottom: 6px;
}

.insight-desc {
    color: #52616B;
    font-size: 13px;
    line-height: 1.4;
}
            
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown("""
    <div class="main-title">
        Saudi Real Estate Rental Analytics Dashboard
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="subtitle">
        An interactive dashboard analyzing real AQAR rental listings across major Saudi cities.
    </div>
""", unsafe_allow_html=True)

# ---------------- DATA ----------------
df = pd.read_csv("data/real_estate.csv")

# Keep only useful columns
df = df[[
    "city", "district", "size", "property_age",
    "bedrooms", "bathrooms", "livingrooms",
    "garage", "furnished", "duplex", "price"
]]

# Rename columns to match the dashboard
df = df.rename(columns={
    "city": "City",
    "district": "District",
    "size": "Area_sqm",
    "property_age": "Property_Age",
    "bedrooms": "Bedrooms",
    "bathrooms": "Bathrooms",
    "livingrooms": "Living_Rooms",
    "garage": "Garage",
    "furnished": "Furnished",
    "duplex": "Duplex",
    "price": "Price_SAR"
})

# Convert numeric columns
numeric_cols = [
    "Area_sqm", "Property_Age", "Bedrooms", "Bathrooms",
    "Living_Rooms", "Garage", "Furnished", "Duplex", "Price_SAR"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove missing values from essential columns
df = df.dropna(subset=["City", "District", "Area_sqm", "Price_SAR"])

# Fill missing numeric values
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Remove unrealistic and extreme values
df = df[df["Area_sqm"] >= 50]
df = df[df["Area_sqm"] <= 2000]

df = df[df["Price_SAR"] >= 5000]
df = df[df["Price_SAR"] <= 1500000]

# Create price per square meter
df["Price_per_sqm"] = df["Price_SAR"] / df["Area_sqm"]

# Remove unrealistic price per sqm values
df = df[df["Price_per_sqm"] >= 10]
df = df[df["Price_per_sqm"] <= 10000]

# Create property type based on available dataset columns
df["Property_Type"] = np.where(df["Duplex"] == 1, "Duplex", "House")

# Clean text columns
df["City"] = df["City"].astype(str).str.strip()
df["District"] = df["District"].astype(str).str.strip()

# Basic city coordinates for map
city_lat = {
    "الرياض": 24.7136,
    "جدة": 21.4858,
    "الدمام": 26.4207,
    "الخبر": 26.2172
}

city_lon = {
    "الرياض": 46.6753,
    "جدة": 39.1925,
    "الدمام": 50.0888,
    "الخبر": 50.1971
}

df["lat"] = df["City"].map(city_lat)
df["lon"] = df["City"].map(city_lon)

# Remove rows without map coordinates
df = df.dropna(subset=["lat", "lon"])

# ---------------- SIDEBAR FILTERS ----------------
st.sidebar.header("Filters")

city_filter = st.sidebar.multiselect(
    "Select City",
    options=df["City"].unique(),
    default=df["City"].unique()
)

district_filter = st.sidebar.multiselect(
    "Select District",
    options=df[df["City"].isin(city_filter)]["District"].unique(),
    default=df[df["City"].isin(city_filter)]["District"].unique()
)

type_filter = st.sidebar.multiselect(
    "Property Type",
    options=df["Property_Type"].unique(),
    default=df["Property_Type"].unique()
)

filtered_df = df[
    (df["City"].isin(city_filter)) &
    (df["District"].isin(district_filter)) &
    (df["Property_Type"].isin(type_filter))
]

if filtered_df.empty:
    st.warning("No data found. Please adjust the filters.")
    st.stop()

# ---------------- KPI ----------------
st.markdown("### Key Market Metrics")

avg_price = int(filtered_df["Price_SAR"].mean())
median_price = int(filtered_df["Price_SAR"].median())
avg_area = int(filtered_df["Area_sqm"].mean())
avg_price_sqm = int(filtered_df["Price_per_sqm"].mean())
total_listings = len(filtered_df)

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "Total Listings",
    f"{total_listings:,}"
)

col2.metric(
    "Avg Rental Price",
    f"{avg_price:,} SAR"
)

col3.metric(
    "Median Rental Price",
    f"{median_price:,} SAR"
)

col4.metric(
    "Avg Area",
    f"{avg_area:,} sqm"
)

col5.metric(
    "Avg Price / sqm",
    f"{avg_price_sqm:,} SAR"
)

# ---------------- INSIGHT BOXES ----------------
st.markdown("### Market Insights")

city_insight = (
    filtered_df
    .groupby("City", as_index=False)
    .agg(
        Avg_Price=("Price_SAR", "mean"),
        Listings=("Price_SAR", "count")
    )
    .sort_values(by="Avg_Price", ascending=False)
)

top_city = city_insight.iloc[0]["City"]
top_city_price = int(city_insight.iloc[0]["Avg_Price"])

district_insight = (
    filtered_df
    .groupby(["City", "District"], as_index=False)
    .agg(
        Avg_Price=("Price_SAR", "mean"),
        Listings=("Price_SAR", "count")
    )
)

district_insight_filtered = district_insight[district_insight["Listings"] >= 3]

if district_insight_filtered.empty:
    district_insight_filtered = district_insight

top_district_row = district_insight_filtered.sort_values(by="Avg_Price", ascending=False).iloc[0]
top_district = f"{top_district_row['City']} - {top_district_row['District']}"
top_district_price = int(top_district_row["Avg_Price"])

most_common_type = filtered_df["Property_Type"].mode()[0]
most_common_type_count = filtered_df["Property_Type"].value_counts().iloc[0]

area_price_corr = filtered_df["Area_sqm"].corr(filtered_df["Price_SAR"])

if pd.isna(area_price_corr):
    corr_label = "Not enough data"
    corr_desc = "The selected filters do not provide enough data to measure the relationship."
elif area_price_corr >= 0.5:
    corr_label = "Strong positive"
    corr_desc = "Larger properties generally show higher rental prices."
elif area_price_corr >= 0.2:
    corr_label = "Moderate positive"
    corr_desc = "Area affects rental price, but location and other factors still matter."
else:
    corr_label = "Weak relationship"
    corr_desc = "Area alone does not explain rental price differences clearly."

insight_col1, insight_col2, insight_col3, insight_col4 = st.columns(4)

with insight_col1:
    st.markdown(f"""
    <div class="insight-card">
        <div class="insight-title">Highest Avg City</div>
        <div class="insight-value">{top_city}</div>
        <div class="insight-desc">Average rental price: {top_city_price:,} SAR</div>
    </div>
    """, unsafe_allow_html=True)

with insight_col2:
    st.markdown(f"""
    <div class="insight-card">
        <div class="insight-title">Highest Avg District</div>
        <div class="insight-value">{top_district}</div>
        <div class="insight-desc">Average rental price: {top_district_price:,} SAR</div>
    </div>
    """, unsafe_allow_html=True)

with insight_col3:
    st.markdown(f"""
    <div class="insight-card">
        <div class="insight-title">Dominant Property Type</div>
        <div class="insight-value">{most_common_type}</div>
        <div class="insight-desc">Number of listings: {most_common_type_count:,}</div>
    </div>
    """, unsafe_allow_html=True)

with insight_col4:
    st.markdown(f"""
    <div class="insight-card">
        <div class="insight-title">Area-Price Relationship</div>
        <div class="insight-value">{corr_label}</div>
        <div class="insight-desc">{corr_desc}</div>
    </div>
    """, unsafe_allow_html=True)
    
# ---------------- CHART 1 ----------------
st.markdown("### Top Districts by Average Rental Price")

district_price = (
    filtered_df
    .groupby(["City", "District"], as_index=False)
    .agg(
        Avg_Price=("Price_SAR", "mean"),
        Avg_Price_per_sqm=("Price_per_sqm", "mean"),
        Listings=("Price_SAR", "count")
    )
)

# نخلي الشارت يعرض الأحياء اللي عندها عدد بيانات معقول
district_price = district_price[district_price["Listings"] >= 3]

district_price["Location"] = district_price["City"] + " - " + district_price["District"]

top_districts = (
    district_price
    .sort_values(by="Avg_Price", ascending=False)
    .head(10)
    .sort_values(by="Avg_Price", ascending=True)
)

fig = px.bar(
    top_districts,
    x="Avg_Price",
    y="Location",
    orientation="h",
    text="Avg_Price",
    hover_data={
        "City": True,
        "District": True,
        "Listings": True,
        "Avg_Price_per_sqm": ":,.0f",
        "Avg_Price": ":,.0f"
    }
)

fig.update_traces(
    marker_color="#102A43",
    texttemplate="%{text:,.0f} SAR",
    textposition="outside"
)

fig.update_layout(
    xaxis_title="Average Rental Price (SAR)",
    yaxis_title="City - District",
    height=550,
    showlegend=False,
    plot_bgcolor="#F7F5EF",
    paper_bgcolor="#F7F5EF",
    font=dict(color="#102A43")
)

st.plotly_chart(fig, use_container_width=True)

# ---------------- CHART 2 ----------------
st.markdown("### Area vs Rental Price")

fig2, ax2 = plt.subplots(figsize=(10, 5))

sns.scatterplot(
    data=filtered_df,
    x="Area_sqm",
    y="Price_SAR",
    hue="Property_Type",
    palette={
        "House": "#102A43",
        "Duplex": "#C9A646"
    },
    ax=ax2
)

ax2.set_title("Relationship Between Property Area and Rental Price", color="#102A43")
ax2.set_xlabel("Area (sqm)")
ax2.set_ylabel("Price (SAR)")
ax2.ticklabel_format(style="plain", axis="y")

ax2.set_facecolor("#F7F5EF")
fig2.patch.set_facecolor("#F7F5EF")

st.pyplot(fig2)

# ---------------- CHART 3 ----------------
st.markdown("### Property Type Distribution")

property_counts = (
    filtered_df["Property_Type"]
    .value_counts()
    .reset_index()
)

property_counts.columns = ["Property_Type", "Count"]

fig3 = px.pie(
    property_counts,
    names="Property_Type",
    values="Count",
    hole=0.55,
    color="Property_Type",
    color_discrete_map={
        "House": "#102A43",
        "Duplex": "#C9A646"
    }
)

fig3.update_traces(
    textposition="inside",
    textinfo="percent+label",
    marker=dict(line=dict(color="#F7F5EF", width=2))
)

fig3.update_layout(
    title="Share of Property Types",
    title_font=dict(size=20, color="#102A43"),
    showlegend=True,
    height=450,
    paper_bgcolor="#F7F5EF",
    plot_bgcolor="#F7F5EF",
    font=dict(color="#102A43"),
    legend_title_text="Property Type"
)

st.plotly_chart(fig3, use_container_width=True)

# ---------------- MAP ----------------
st.markdown("### City-Level Rental Market Map")

city_map = (
    filtered_df
    .groupby(["City", "lat", "lon"], as_index=False)
    .agg(
        Listings=("Price_SAR", "count"),
        Avg_Rental_Price=("Price_SAR", "mean"),
        Avg_Price_per_sqm=("Price_per_sqm", "mean")
    )
)

fig_map = px.scatter_mapbox(
    city_map,
    lat="lat",
    lon="lon",
    size="Listings",
    color="Avg_Rental_Price",
    hover_name="City",
    hover_data={
        "Listings": True,
        "Avg_Rental_Price": ":,.0f",
        "Avg_Price_per_sqm": ":,.0f",
        "lat": False,
        "lon": False
    },
    zoom=4.5,
    height=500,
    color_continuous_scale=["#C9A646", "#102A43"]
)

fig_map.update_layout(
    mapbox_style="open-street-map",
    paper_bgcolor="#F7F5EF",
    plot_bgcolor="#F7F5EF",
    font=dict(color="#102A43"),
    margin=dict(l=0, r=0, t=20, b=0),
    coloraxis_colorbar=dict(
        title="Avg Price"
    )
)

st.plotly_chart(fig_map, use_container_width=True)

# ---------------- ML MODEL ----------------
st.markdown("### Rental Price Prediction Model")

model_df = df.copy()

model_features = [
    "Area_sqm",
    "Property_Age",
    "Bedrooms",
    "Bathrooms",
    "Living_Rooms",
    "Garage",
    "Furnished",
    "Duplex",
    "City",
    "District",
    "Property_Type"
]

target = "Price_SAR"

model_df = model_df[model_features + [target]].copy()

# Numeric columns
numeric_model_cols = [
    "Area_sqm",
    "Property_Age",
    "Bedrooms",
    "Bathrooms",
    "Living_Rooms",
    "Garage",
    "Furnished",
    "Duplex"
]

# Clean numeric columns
for col in numeric_model_cols:
    model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
    model_df[col] = model_df[col].fillna(model_df[col].median())

# Clean text columns
text_model_cols = ["City", "District", "Property_Type"]

for col in text_model_cols:
    model_df[col] = model_df[col].astype(str).fillna("Unknown")

# Remove rows with missing target
model_df[target] = pd.to_numeric(model_df[target], errors="coerce")
model_df = model_df.dropna(subset=[target])

X = model_df[model_features]
y = model_df[target]

# Convert categorical columns into numeric columns
X_encoded = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.2,
    random_state=42
)

# Train Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

linear_pred = linear_model.predict(X_test)

linear_mae = mean_absolute_error(y_test, linear_pred)
linear_r2 = r2_score(y_test, linear_pred)

# Train Random Forest
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

# Model comparison table
model_results = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "MAE": [linear_mae, rf_mae],
    "R2 Score": [linear_r2, rf_r2]
})

model_results["MAE"] = model_results["MAE"].round(0).astype(int)
model_results["R2 Score"] = model_results["R2 Score"].round(3)

st.markdown("#### Model Performance Comparison")

st.dataframe(
    model_results,
    use_container_width=True
)

# Show best model based on MAE
best_model_name = model_results.sort_values("MAE").iloc[0]["Model"]

st.info(
    f"The best performing model based on MAE is: {best_model_name}"
)

# ---------------- FEATURE IMPORTANCE ----------------
st.markdown("#### Key Factors Affecting Rental Price")

feature_importance = pd.DataFrame({
    "Feature": X_encoded.columns,
    "Importance": rf_model.feature_importances_
})

# Group encoded categorical variables into readable main factors
def group_feature_name(feature):
    if feature.startswith("City_"):
        return "City"
    elif feature.startswith("District_"):
        return "District"
    elif feature.startswith("Property_Type_"):
        return "Property Type"
    else:
        return feature

feature_importance["Main_Factor"] = feature_importance["Feature"].apply(group_feature_name)

factor_importance = (
    feature_importance
    .groupby("Main_Factor", as_index=False)["Importance"]
    .sum()
    .sort_values(by="Importance", ascending=False)
)

# Rename technical column names to cleaner labels
factor_labels = {
    "Area_sqm": "Area",
    "Property_Age": "Property Age",
    "Bedrooms": "Bedrooms",
    "Bathrooms": "Bathrooms",
    "Living_Rooms": "Living Rooms",
    "Garage": "Garage",
    "Furnished": "Furnished",
    "Duplex": "Duplex",
    "City": "City",
    "District": "District",
    "Property Type": "Property Type"
}

factor_importance["Main_Factor"] = factor_importance["Main_Factor"].replace(factor_labels)

top_factors = factor_importance.head(10).sort_values(by="Importance", ascending=True)

fig_importance = px.bar(
    top_factors,
    x="Importance",
    y="Main_Factor",
    orientation="h",
    text="Importance",
    color_discrete_sequence=["#102A43"]
)

fig_importance.update_traces(
    texttemplate="%{text:.3f}",
    textposition="outside"
)

fig_importance.update_layout(
    title="Top Factors Influencing Rental Price",
    title_font=dict(size=20, color="#102A43"),
    xaxis_title="Importance Score",
    yaxis_title=None,
    height=450,
    showlegend=False,
    plot_bgcolor="#F7F5EF",
    paper_bgcolor="#F7F5EF",
    font=dict(color="#102A43"),
    margin=dict(l=20, r=40, t=60, b=40)
)

st.plotly_chart(fig_importance, use_container_width=True)

top_factor_name = factor_importance.iloc[0]["Main_Factor"]
top_factor_score = factor_importance.iloc[0]["Importance"]

st.info(
    f"The most influential factor in the Random Forest model is {top_factor_name}, "
    f"with an importance score of {top_factor_score:.3f}."
)

# Use Random Forest for final prediction
final_model = rf_model

st.markdown("#### Try a Rental Price Prediction")

city_input = st.selectbox(
    "Select City",
    sorted(df["City"].unique())
)

district_options = sorted(df[df["City"] == city_input]["District"].unique())

district_input = st.selectbox(
    "Select District",
    district_options
)

area_input = st.slider(
    "Select Area (sqm)",
    int(df["Area_sqm"].min()),
    int(df["Area_sqm"].max()),
    250
)

age_input = st.slider(
    "Property Age",
    int(df["Property_Age"].min()),
    int(df["Property_Age"].max()),
    5
)

bedrooms_input = st.slider(
    "Bedrooms",
    int(df["Bedrooms"].min()),
    int(df["Bedrooms"].max()),
    4
)

bathrooms_input = st.slider(
    "Bathrooms",
    int(df["Bathrooms"].min()),
    int(df["Bathrooms"].max()),
    3
)

livingrooms_input = st.slider(
    "Living Rooms",
    int(df["Living_Rooms"].min()),
    int(df["Living_Rooms"].max()),
    1
)

type_input = st.selectbox(
    "Property Type",
    sorted(df["Property_Type"].unique())
)

garage_input = st.selectbox("Garage", [0, 1])
furnished_input = st.selectbox("Furnished", [0, 1])
duplex_input = 1 if type_input == "Duplex" else 0

input_data = pd.DataFrame({
    "Area_sqm": [area_input],
    "Property_Age": [age_input],
    "Bedrooms": [bedrooms_input],
    "Bathrooms": [bathrooms_input],
    "Living_Rooms": [livingrooms_input],
    "Garage": [garage_input],
    "Furnished": [furnished_input],
    "Duplex": [duplex_input],
    "City": [city_input],
    "District": [district_input],
    "Property_Type": [type_input]
})

input_encoded = pd.get_dummies(input_data)
input_encoded = input_encoded.reindex(columns=X_encoded.columns, fill_value=0)

predicted_price = final_model.predict(input_encoded)[0]

st.success(f"Predicted Rental Price: {int(predicted_price):,} SAR")

# ---------------- DOWNLOAD BUTTON ----------------
st.markdown("### Download Data")

csv = filtered_df.to_csv(index=False).encode('utf-8')

st.download_button(
    label="Download Dataset",
    data=csv,
    file_name="jeddah_real_estate_data.csv",
    mime="text/csv"
)

# ---------------- TABLE ----------------

st.markdown("### Data Table")

st.write("Number of rows:", len(filtered_df))

if filtered_df.empty:
    st.warning("No data found. The filters may be hiding all rows.")
else:
    st.dataframe(filtered_df, use_container_width=True)