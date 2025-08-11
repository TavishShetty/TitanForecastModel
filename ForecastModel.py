import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from datetime import datetime

st.set_page_config(page_title="TITAN Monthly Forecast", layout="wide")
st.title("📈:green[TITAN] :blue[Sales Forecast]")

uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file, sheet_name="Pivot")

        df.columns = df.columns.str.strip().str.lower()

        column_map = {
            'theme code': 'theme_code',
            'category': 'category',
            'total orders': 'total_orders',
            'sold': 'sold',
            'stock in showroom': 'stock_in_showroom',
            '22-23': '22_23',
            '23-24': '23_24',
            '24-25': '24_25',
            '25-26': '25_26'
        }
        df.rename(columns=column_map, inplace=True)

        year_columns = ['22_23', '23_24', '24_25', '25_26']
        df = df.dropna(subset=year_columns, how='all')

        # Detect number of completed months in current year (assumed to be '25_26')
        current_month = datetime.now().month
        current_year_completed_months = current_month if current_month <= 12 else 12

        for year in year_columns:
            if year == '25_26':
                df[year] = pd.to_numeric(df[year], errors='coerce') / current_year_completed_months
            else:
                df[year] = pd.to_numeric(df[year], errors='coerce') / 12

        forecast_list = []

        for _, row in df.iterrows():
            theme_code = row.get('theme_code', 'Unknown')
            category = row.get('category', 'Unknown')
            sales = row[year_columns].values.astype(float)

            # Linear Regression forecast
            X = np.array(range(len(sales))).reshape(-1, 1)
            y = sales.reshape(-1, 1)
            model = LinearRegression()
            model.fit(X, y)
            forecast = model.predict([[len(sales)]])[0][0]

            # Total orders and sold for efficiency ratio
            total_orders = pd.to_numeric(row.get("total_orders", 0), errors='coerce')
            sold = pd.to_numeric(row.get("sold", 0), errors='coerce')

            if pd.notna(total_orders) and total_orders > 0 and pd.notna(sold):
                efficiency_ratio = sold / total_orders
            else:
                efficiency_ratio = 1  # default if data missing

            adjusted_forecast = forecast * efficiency_ratio
            adjusted_forecast = max(0, round(adjusted_forecast))

            stock = pd.to_numeric(row.get('stock_in_showroom', 0), errors='coerce')
            stock = 0 if pd.isna(stock) else round(stock)

            recommendation = "Stock Up" if adjusted_forecast > stock else "Sufficient Stock"

            forecast_list.append({
                "Theme Code": theme_code,
                "Category": category,
                "Forecast (Next Month)": adjusted_forecast,
                "Total Orders": total_orders,
                "SOLD": sold,
                "Stock in Showroom": stock,
                "Recommendation": recommendation
            })

        forecast_df = pd.DataFrame(forecast_list)
        display_df = forecast_df.drop(columns=["Recommendation"])

        st.subheader("📊 Monthly Forecast for Next Month")
        st.dataframe(display_df)

        st.subheader("🔎 Filter by Category")
        categories = forecast_df['Category'].dropna().unique().tolist()
        selected_categories = st.multiselect("Select Category", categories, default=categories)
        filtered_df = forecast_df[forecast_df['Category'].isin(selected_categories)]
        display_filtered_df = filtered_df.drop(columns=["Recommendation"])
        st.dataframe(display_filtered_df)

        st.subheader("⬇️ Download Forecast")
        sorted_df = display_filtered_df.sort_values(by="Forecast (Next Month)", ascending=False)
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            sorted_df.to_excel(writer, index=False, sheet_name='Forecast')
        st.download_button(
            label="📥 Download Forecast as Excel",
            data=buffer.getvalue(),
            file_name="Titan_Forecast.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.subheader("📉 Top Forecasted Products")
        top_n = st.slider("Select number of products", 5, 30, 10)
        chart_data = filtered_df.sort_values(by="Forecast (Next Month)", ascending=False).head(top_n)

        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=chart_data,
            x="Forecast (Next Month)",
            y="Theme Code",
            hue="Category",
            dodge=False,
            palette="viridis"
        )
        plt.title("Top Products Forecasted for Next Month")
        plt.xlabel("Forecasted Sales")
        plt.ylabel("Theme Code")
        st.pyplot(plt.gcf())

    except Exception as e:
        st.error(f"❌ Error processing the file: {e}")