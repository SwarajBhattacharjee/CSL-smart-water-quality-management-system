import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import re

# ------------------- PAGE CONFIG -------------------
st.set_page_config(
    page_title="Water Analysis (River & Groundwater)",
    layout="wide"
)

# ------------------- LOADING DATA -------------------
@st.cache_data
def load_river_data():
    """Loads the River CSV file."""
    df_river = pd.read_csv("data/river.csv")  # Adjust path if needed
    return df_river

@st.cache_data
def load_groundwater_data():
    """Loads the Groundwater Excel file."""
    df_gw = pd.read_excel("data/Modified_filled.xlsx")  # Adjust path if needed
    return df_gw

# ------------------- RIVER: CREATE MEAN COLUMNS (Min/Max => Mean) -------------------
def create_mean_columns_river(df):
    """
    For each pair like {prefix}_Min_{YYYY} and {prefix}_Max_{YYYY},
    create {prefix}_Mean_{YYYY} = (Min + Max)/2.
    """
    new_df = df.copy()

    pattern = r"^(.+?)_(Min|Max)_(\d{4})$"
    pairs = {}
    for col in new_df.columns:
        match = re.match(pattern, col)
        if match:
            prefix, which, year = match.groups()
            key = (prefix, year)
            if key not in pairs:
                pairs[key] = {"Min": None, "Max": None}
            pairs[key][which] = col

    for (prefix, year), mm in pairs.items():
        min_col = mm.get("Min")
        max_col = mm.get("Max")
        if min_col and max_col:
            mean_col = f"{prefix}_Mean_{year}"
            new_df[min_col] = pd.to_numeric(new_df[min_col], errors='coerce')
            new_df[max_col] = pd.to_numeric(new_df[max_col], errors='coerce')
            new_df[mean_col] = (new_df[min_col] + new_df[max_col]) / 2.0
    return new_df

# ------------------- GROUNDWATER PREPARATION -------------------
def prepare_groundwater_data(df):
    """
    Renames 'Years' -> 'Year' and ensures water quality columns are numeric.
    """
    new_df = df.copy()
    if "Years" in new_df.columns:
        new_df.rename(columns={"Years": "Year"}, inplace=True)
    else:
        st.warning("'Years' column not found in groundwater data.")

    possible_params = ["pH", "Conductivity", "Temperature", "BOD", "Fecal_Coliform"]
    for col in possible_params:
        if col in new_df.columns:
            new_df[col] = pd.to_numeric(new_df[col], errors='coerce')

    return new_df

# ------------------- POTABILITY CHECK: GROUNDWATER / GENERAL -------------------
def determine_potability_by_year(df, year_col="Year"):
    """
    For datasets that have a 'Year' column + direct numeric columns:
    (pH, Conductivity, Temperature, BOD, Fecal_Coliform)
    """
    if year_col not in df.columns:
        return pd.DataFrame()

    rules = {
        "pH": lambda x: 6.5 <= x <= 8.5,
        "Conductivity": lambda x: x <= 2000,
        "Temperature": lambda x: x <= 35,
        "BOD": lambda x: x <= 5,
        "Fecal_Coliform": lambda x: x <= 1,
    }

    grouped = df.groupby(year_col).mean(numeric_only=True).reset_index()
    result_rows = []
    for _, row in grouped.iterrows():
        y = int(row[year_col])
        is_potable = True
        measures = {}

        for param, condition in rules.items():
            if param in row and not pd.isna(row[param]):
                val = row[param]
                measures[param] = val
                if not condition(val):
                    is_potable = False
            else:
                is_potable = False

        result_rows.append({
            "Year": y,
            "Potable": "Yes" if is_potable else "No",
            **measures
        })

    return pd.DataFrame(result_rows).sort_values("Year")

# ------------------- POTABILITY CHECK: RIVER (USING _Mean_YYYY COLUMNS) -------------------
def determine_river_potability_from_mean_cols(df):
    """
    For River data:
      - Identify columns like pH_Mean_2023
      - Compute average across rows for each param-year
      - Check potability thresholds
    """
    rules = {
        "pH": lambda x: 6.5 <= x <= 8.5,
        "Conductivity": lambda x: x <= 2000,
        "Temperature": lambda x: x <= 35,
        "BOD": lambda x: x <= 5,
        "Fecal_Coliform": lambda x: x <= 1
    }

    pattern = r"^(.+?)_Mean_(\d{4})$"
    year_map = {}
    for col in df.columns:
        m = re.match(pattern, col)
        if m:
            param, year_str = m.groups()
            df[col] = pd.to_numeric(df[col], errors='coerce')
            col_mean = df[col].mean(skipna=True)
            yr = int(year_str)
            if yr not in year_map:
                year_map[yr] = {}
            year_map[yr][param] = col_mean

    result_rows = []
    for y in sorted(year_map.keys()):
        param_vals = year_map[y]
        is_potable = True
        row_dict = {"Year": y}
        for param, fn in rules.items():
            if param in param_vals and not pd.isna(param_vals[param]):
                val = param_vals[param]
                row_dict[param] = val
                if not fn(val):
                    is_potable = False
            else:
                is_potable = False
        row_dict["Potable"] = "Yes" if is_potable else "No"
        result_rows.append(row_dict)

    return pd.DataFrame(result_rows)

# ------------------- YEARLY ANALYSIS HELPER (Groundwater, etc.) -------------------
def get_yearly_data(df, col_name, year_col="Year"):
    """
    For data that truly has a 'Year' column, group by year => [Year, Value].
    """
    if year_col not in df.columns or col_name not in df.columns:
        return pd.DataFrame(columns=["Year", "Value"])

    grouped = (
        df.dropna(subset=[col_name, year_col])
          .groupby(year_col)[col_name]
          .mean()
          .reset_index(name="Value")
    )
    grouped = grouped.sort_values(year_col)
    grouped[year_col] = grouped[year_col].astype(int)
    return grouped.rename(columns={year_col: "Year"})

# ------------------- FORECAST HELPERS -------------------
def forecast_next_5_years(yearly_df, model_name="Decision Tree"):
    """
    For a [Year, Value] DataFrame, forecast next 5 years using user-chosen model.
    """
    if yearly_df.empty:
        return pd.DataFrame(columns=["Year", "Value"])

    X = yearly_df[["Year"]].values
    y = yearly_df["Value"].values

    if model_name == "Decision Tree":
        model = DecisionTreeRegressor(random_state=42)
    elif model_name == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_name == "Linear Regression":
        model = LinearRegression()
    else:
        st.warning("Unknown model selected, defaulting to Decision Tree.")
        model = DecisionTreeRegressor(random_state=42)

    model.fit(X, y)

    last_year = int(yearly_df["Year"].max())
    future_years = np.arange(last_year + 1, last_year + 6)
    future_values = model.predict(future_years.reshape(-1, 1))

    future_df = pd.DataFrame({"Year": future_years, "Value": future_values})
    return future_df

# ------------------- NEW HELPER FOR RIVER FORECAST FROM MEAN COLS -------------------
def get_river_yearly_data_from_mean_cols(df, prefix):
    """
    Parse columns like {prefix}_Mean_YYYY => (YYYY, average across rows).
    Return a DataFrame with [Year, Value].
    """
    pattern = rf"^{prefix}_Mean_(\d{{4}})$"
    data_list = []
    for col in df.columns:
        m = re.match(pattern, col)
        if m:
            year_str = m.group(1)
            df[col] = pd.to_numeric(df[col], errors="coerce")
            avg_val = df[col].mean(skipna=True)
            data_list.append({"Year": int(year_str), "Value": avg_val})
    out_df = pd.DataFrame(data_list)
    out_df = out_df.dropna(subset=["Value"]).sort_values("Year")
    return out_df

def get_river_prefixes(df):
    """
    Scan columns for patterns like prefix_Mean_YYYY
    and collect unique 'prefix' names.
    """
    pattern = r"^(.+?)_Mean_(\d{4})$"
    prefixes = set()
    for col in df.columns:
        m = re.match(pattern, col)
        if m:
            prefix = m.group(1)
            prefixes.add(prefix)
    return sorted(prefixes)

# ------------------- LOAD & PREPARE DATA -------------------
df_river_raw = load_river_data()
df_river = create_mean_columns_river(df_river_raw)

df_ground_raw = load_groundwater_data()
df_ground = prepare_groundwater_data(df_ground_raw)

# Potability
ground_potability = determine_potability_by_year(df_ground, year_col="Year")
river_potability = determine_river_potability_from_mean_cols(df_river)

# Identify numeric columns for existing (non-mean) usage
def get_numeric_columns_excluding_year(df, year_col="Year"):
    numeric_cols = []
    for c in df.columns:
        if c != year_col and pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)
    return numeric_cols

river_cols = get_numeric_columns_excluding_year(df_river, year_col="Year")
ground_cols = get_numeric_columns_excluding_year(df_ground, year_col="Year")
common_cols = sorted(list(set(river_cols) & set(ground_cols)))

# ------------------- STREAMLIT TABS -------------------
tab_river, tab_ground, tab_compare = st.tabs(["River Data", "Groundwater Data", "Comparative Analysis"])

# ------------------- RIVER TAB -------------------
with tab_river:
    st.header("River Data Analysis")

    # 1) OVERVIEW
    st.subheader("1) Overview")
    st.write("**Sample (first 5 rows):**")
    st.write(df_river.head(5))
    st.write(f"**Shape**: {df_river.shape[0]} rows x {df_river.shape[1]} columns")
    st.write("**Missing Values:**")
    missing_counts = df_river.isnull().sum()
    st.write(missing_counts[missing_counts > 0].sort_values(ascending=False))

    # 2) POTABILITY
    st.subheader("2) Potability by Year (Based on Min/Max => Mean Columns)")
    if river_potability.empty:
        st.warning("No potability data for River. Possibly no _Mean_YYYY columns found.")
    else:
        st.dataframe(river_potability)

    # 3) YEARLY ANALYSIS (UNCHANGED)
    st.subheader("3) Yearly Analysis for Selected Parameter")
    st.markdown(
        "This still expects a 'Year' column in the dataset, if you have one. "
        "Otherwise, it may show empty results."
    )
    if not river_cols:
        st.warning("No numeric columns found in River data (excluding 'Year').")
    else:
        sel_river_col = st.selectbox("Select a River parameter", river_cols, key="river_parameter")
        river_yearly_df = get_yearly_data(df_river, sel_river_col, year_col="Year")
        if river_yearly_df.empty:
            st.write("No data for that parameter or missing 'Year'.")
        else:
            st.dataframe(river_yearly_df)

            fig, ax = plt.subplots(figsize=(5,3))
            ax.plot(river_yearly_df["Year"], river_yearly_df["Value"], marker="o")
            ax.set_title(f"River: {sel_river_col} by Year")
            ax.set_xlabel("Year")
            ax.set_ylabel("Average Value")
            st.pyplot(fig)

    # 4) 5-YEAR FORECAST (UPDATED TO USE _Mean_YYYY COLUMNS FOR RIVER)
    st.subheader("4) 5-Year Forecast (Using Mean Columns)")
    st.markdown(
        "Select a **parameter prefix** from `_Mean_YYYY` columns in river.csv. "
        "We'll build [Year, Value] from those columns and forecast the next 5 years."
    )

    # Gather available prefixes: e.g. pH, Conductivity, etc.
    river_prefixes = get_river_prefixes(df_river)
    if not river_prefixes:
        st.warning("No 'prefix_Mean_YYYY' columns found in River data. Forecast not available.")
    else:
        sel_prefix = st.selectbox("Select parameter prefix", river_prefixes, key="river_prefix_forecast")
        model_choice = st.selectbox(
            "Select forecast model",
            ["Decision Tree", "Random Forest", "Linear Regression"],
            key="river_model_forecast"
        )

        # Build [Year, Value] from the user-chosen prefix
        prefix_data = get_river_yearly_data_from_mean_cols(df_river, sel_prefix)
        if prefix_data.empty:
            st.write("No data found for that prefix.")
        else:
            # Forecast next 5 years
            future_df = forecast_next_5_years(prefix_data, model_name=model_choice)
            combined_df = pd.concat([prefix_data, future_df]).reset_index(drop=True)

            st.write("**Forecast Results (Next 5 Years)**:")
            st.dataframe(future_df)

            csv_data = future_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Forecast Data (CSV)",
                data=csv_data,
                file_name=f"{sel_prefix}_river_mean_forecast.csv",
                mime="text/csv"
            )

            # Plot
            fig, ax = plt.subplots(figsize=(5,3))
            last_hist_year = prefix_data["Year"].max()
            hist_mask = combined_df["Year"] <= last_hist_year
            fut_mask = combined_df["Year"] > last_hist_year

            ax.plot(
                combined_df.loc[hist_mask, "Year"],
                combined_df.loc[hist_mask, "Value"],
                marker="o", label="Historical"
            )
            ax.plot(
                combined_df.loc[fut_mask, "Year"],
                combined_df.loc[fut_mask, "Value"],
                marker="o", label="Forecast"
            )
            ax.set_title(f"{sel_prefix} Forecast ({model_choice}) - River")
            ax.set_xlabel("Year")
            ax.set_ylabel("Value")
            ax.legend()
            st.pyplot(fig)

# ------------------- GROUNDWATER TAB -------------------
with tab_ground:
    st.header("Groundwater Data Analysis")
    # (Unchanged from previous code)

    st.subheader("1) Overview")
    st.write("**Sample (first 5 rows):**")
    st.write(df_ground.head(5))
    st.write(f"**Shape**: {df_ground.shape[0]} rows x {df_ground.shape[1]} columns")
    st.write("**Missing Values:**")
    missing_counts_gw = df_ground.isnull().sum()
    st.write(missing_counts_gw[missing_counts_gw > 0].sort_values(ascending=False))

    st.subheader("2) Potability by Year")
    if ground_potability.empty:
        st.warning("No potability data for Groundwater. Possibly missing columns or 'Year'.")
    else:
        st.dataframe(ground_potability)

    st.subheader("3) Yearly Analysis for Selected Parameter")
    if not ground_cols:
        st.warning("No numeric columns found in Groundwater data (excluding 'Year').")
    else:
        gw_sel_col = st.selectbox("Select a Groundwater parameter", ground_cols, key="gw_parameter")
        gw_yearly_df = get_yearly_data(df_ground, gw_sel_col, year_col="Year")
        if gw_yearly_df.empty:
            st.write("No data for that parameter or missing 'Year'.")
        else:
            st.dataframe(gw_yearly_df)

            fig, ax = plt.subplots(figsize=(5,3))
            ax.plot(gw_yearly_df["Year"], gw_yearly_df["Value"], marker="o")
            ax.set_title(f"Groundwater: {gw_sel_col} by Year")
            ax.set_xlabel("Year")
            ax.set_ylabel("Average Value")
            st.pyplot(fig)

    st.subheader("4) 5-Year Forecast")
    if not ground_cols:
        st.warning("No numeric columns to forecast in Groundwater data.")
    else:
        gw_fc_col = st.selectbox("Select parameter to forecast", ground_cols, key="gw_forecast")
        fc_model_gw = st.selectbox("Select forecast model", ["Decision Tree", "Random Forest", "Linear Regression"], key="ground_model")

        gw_fc_df = get_yearly_data(df_ground, gw_fc_col, year_col="Year")
        if gw_fc_df.empty:
            st.write("No data to forecast for that parameter.")
        else:
            future_gw_df = forecast_next_5_years(gw_fc_df, model_name=fc_model_gw)
            combined_gw_df = pd.concat([gw_fc_df, future_gw_df]).reset_index(drop=True)

            st.write("**Forecast Results (Next 5 Years)**:")
            st.dataframe(future_gw_df)

            csv_gw = future_gw_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Forecast Data (CSV)",
                data=csv_gw,
                file_name=f"{gw_fc_col}_groundwater_forecast.csv",
                mime="text/csv"
            )

            fig, ax = plt.subplots(figsize=(5,3))
            last_hist_year = gw_fc_df["Year"].max()
            hist_mask = combined_gw_df["Year"] <= last_hist_year
            fut_mask = combined_gw_df["Year"] > last_hist_year

            ax.plot(
                combined_gw_df.loc[hist_mask, "Year"],
                combined_gw_df.loc[hist_mask, "Value"],
                marker="o", label="Historical"
            )
            ax.plot(
                combined_gw_df.loc[fut_mask, "Year"],
                combined_gw_df.loc[fut_mask, "Value"],
                marker="o", label="Forecast"
            )
            ax.set_title(f"{gw_fc_col} Forecast ({fc_model_gw}) - Groundwater")
            ax.set_xlabel("Year")
            ax.set_ylabel("Value")
            ax.legend()
            st.pyplot(fig)

# ------------------- COMPARATIVE TAB -------------------
with tab_compare:
    st.header("Comparative Analysis (River vs Groundwater)")
    if not common_cols:
        st.warning("No common numeric columns between River & Groundwater data.")
    else:
        st.write("Common numeric columns:", common_cols)

        # 1) Yearly Comparison
        st.subheader("1) Comparative Yearly Analysis")
        comp_col = st.selectbox("Select a common parameter to compare", common_cols, key="comp_col")

        river_comp_df = get_yearly_data(df_river, comp_col, year_col="Year")
        ground_comp_df = get_yearly_data(df_ground, comp_col, year_col="Year")

        if river_comp_df.empty and ground_comp_df.empty:
            st.write("No data in either dataset for that parameter.")
        else:
            rename_river = river_comp_df.rename(columns={"Value": "RiverValue"})
            rename_gw = ground_comp_df.rename(columns={"Value": "GroundValue"})
            merged = pd.merge(rename_river, rename_gw, on="Year", how="outer").sort_values("Year")

            st.write("**Side-by-side**:")
            st.dataframe(merged)

            fig, ax = plt.subplots(figsize=(5,3))
            if not rename_river.empty:
                ax.plot(rename_river["Year"], rename_river["RiverValue"], marker="o", label="River")
            if not rename_gw.empty:
                ax.plot(rename_gw["Year"], rename_gw["GroundValue"], marker="o", label="Groundwater")
            ax.set_title(f"Comparative: {comp_col} by Year")
            ax.set_xlabel("Year")
            ax.set_ylabel("Average Value")
            ax.legend()
            st.pyplot(fig)

        # 2) Comparative Forecast
        st.subheader("2) Comparative 5-Year Forecast")
        st.markdown("Forecast each dataset separately, then plot together.")

        comp_model_riv = st.selectbox("Select forecast model (River)", ["Decision Tree", "Random Forest", "Linear Regression"], key="comp_model_river")
        comp_model_gw = st.selectbox("Select forecast model (Groundwater)", ["Decision Tree", "Random Forest", "Linear Regression"], key="comp_model_ground")

        river_future = forecast_next_5_years(river_comp_df, model_name=comp_model_riv)
        ground_future = forecast_next_5_years(ground_comp_df, model_name=comp_model_gw)

        combined_river = pd.concat([river_comp_df, river_future])
        combined_ground = pd.concat([ground_comp_df, ground_future])

        st.write("**River Forecast**:")
        st.dataframe(river_future)
        csv_riv = river_future.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download River Forecast CSV",
            data=csv_riv,
            file_name=f"{comp_col}_river_forecast.csv",
            mime="text/csv"
        )

        st.write("**Groundwater Forecast**:")
        st.dataframe(ground_future)
        csv_gnd = ground_future.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Groundwater Forecast CSV",
            data=csv_gnd,
            file_name=f"{comp_col}_ground_forecast.csv",
            mime="text/csv"
        )

        st.write("**Combined Forecast Plot**")
        fig, ax = plt.subplots(figsize=(5,3))

        # Plot River
        if not combined_river.empty:
            if not river_comp_df.empty:
                last_riv_hist = river_comp_df["Year"].max()
                riv_hist_mask = combined_river["Year"] <= last_riv_hist
                riv_fut_mask = combined_river["Year"] > last_riv_hist
                ax.plot(combined_river.loc[riv_hist_mask, "Year"],
                        combined_river.loc[riv_hist_mask, "Value"],
                        marker="o", label="River (Hist)")
                ax.plot(combined_river.loc[riv_fut_mask, "Year"],
                        combined_river.loc[riv_fut_mask, "Value"],
                        marker="o", label="River (Forecast)")
            else:
                ax.plot(combined_river["Year"], combined_river["Value"], marker="o", label="River (Forecast)")

        # Plot Ground
        if not combined_ground.empty:
            if not ground_comp_df.empty:
                last_gnd_hist = ground_comp_df["Year"].max()
                gnd_hist_mask = combined_ground["Year"] <= last_gnd_hist
                gnd_fut_mask = combined_ground["Year"] > last_gnd_hist
                ax.plot(combined_ground.loc[gnd_hist_mask, "Year"],
                        combined_ground.loc[gnd_hist_mask, "Value"],
                        marker="s", label="Ground (Hist)")
                ax.plot(combined_ground.loc[gnd_fut_mask, "Year"],
                        combined_ground.loc[gnd_fut_mask, "Value"],
                        marker="s", label="Ground (Forecast)")
            else:
                ax.plot(combined_ground["Year"], combined_ground["Value"], marker="s", label="Ground (Forecast)")

        ax.set_title(f"Comparative Forecast: {comp_col}")
        ax.set_xlabel("Year")
        ax.set_ylabel("Value")
        ax.legend()
        st.pyplot(fig)

# -------------- END OF APP --------------
st.markdown("---")
st.markdown("**Note**: Only the River forecast step has been updated to use `_Mean_YYYY` columns for forecasting.")
