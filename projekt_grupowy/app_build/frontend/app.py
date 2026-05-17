"""
Streamlit frontend dla systemu predykcji sprzedaży.

Komunikuje się z backendem FastAPI przez REST API.

Strony:
  1. Dashboard ogólny - metryki + lista kategorii + wykresy
  2. Drill-down kategoria - prognoza tygodniowa + trend + produkty
  3. Upload CSV - rekomendacje dla magazynu klienta
"""
import os
import io
from datetime import datetime

import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ─── Konfiguracja ───
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Sales Forecast — Olist",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Helpery API ───

@st.cache_data(ttl=300)
def api_get(endpoint: str):
    """GET request do API z cachowaniem."""
    try:
        r = requests.get(f"{API_URL}{endpoint}", timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error(f"❌ Nie można połączyć się z API ({API_URL}). Czy backend działa?")
        st.stop()
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ API zwróciło błąd: {e.response.status_code} {e.response.text}")
        return None


def api_post_file(endpoint: str, file_bytes: bytes, filename: str):
    """POST plik do API."""
    try:
        files = {"file": (filename, file_bytes, "text/csv")}
        r = requests.post(f"{API_URL}{endpoint}", files=files, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ Błąd API: {e.response.json().get('detail', e.response.text)}")
        return None
    except requests.exceptions.ConnectionError:
        st.error(f"❌ Nie można połączyć się z API ({API_URL})")
        return None


# ─── Health check ───

health = api_get("/")
if not health or not health.get("artifacts_loaded"):
    st.error("⚠ Backend działa ale artefakty nie zostały wczytane.")
    st.info(f"Sprawdź czy folder `artifacts/` zawiera pliki CSV z notebooka.")
    st.stop()


# ─── Sidebar nawigacja ───

st.sidebar.title("📊 Sales Forecast")
st.sidebar.caption("System predykcji sprzedaży e-commerce")

page = st.sidebar.radio(
    "Wybierz widok:",
    ["🏠 Dashboard", "🔍 Szczegóły kategorii", "📤 Upload magazynu (CSV)"],
)

st.sidebar.divider()
st.sidebar.metric("Kategorii w systemie", health.get("categories_count", 0))
st.sidebar.caption(f"API: `{API_URL}`")


# ═══════════════════════════════════════════════════════════════════
# STRONA 1: DASHBOARD
# ═══════════════════════════════════════════════════════════════════

if page == "🏠 Dashboard":
    st.title("📊 Dashboard — przegląd systemu")
    st.caption("Predykcja sprzedaży na 4 tygodnie + rekomendowane stany magazynowe")

    # ── Metryki modelu ──
    metrics = api_get("/metrics")
    if metrics:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Model", metrics.get("model", "XGBoost"))
        col2.metric("RMSE (Test)", f"{float(metrics.get('RMSE', 0)):.2f}")
        col3.metric("MAE (Test)", f"{float(metrics.get('MAE', 0)):.2f}")
        col4.metric("MAPE (Test)", f"{float(metrics.get('MAPE', 0)):.1f}%")

    st.divider()

    # ── Lista kategorii ──
    categories = api_get("/categories")
    if not categories:
        st.warning("Brak danych kategorii")
        st.stop()

    inv_df = pd.DataFrame(categories)

    # KPI strip
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Liczba kategorii", len(inv_df))
    col2.metric("Sumaryczna prognoza (4 tyg)", f"{inv_df['forecast_total_4w'].sum():,.0f} szt")
    col3.metric("Sumaryczny Safety Stock", f"{inv_df['safety_stock'].sum():,.0f} szt")
    col4.metric("Średni RMSE per kat.", f"{inv_df.get('model_rmse', pd.Series([0])).mean():.1f}")

    st.divider()

    # ── Trendy ──
    st.subheader("📈 Rozkład trendów")
    trend_counts = inv_df["trend"].value_counts().reset_index()
    trend_counts.columns = ["trend", "count"]

    col1, col2 = st.columns([1, 2])

    with col1:
        fig_trend = px.pie(
            trend_counts, values="count", names="trend",
            color="trend",
            color_discrete_map={
                "rosnący 📈": "#2ecc71",
                "stabilny ➡️": "#3498db",
                "spadający 📉": "#e74c3c",
                "b/d": "#95a5a6",
            },
        )
        fig_trend.update_layout(height=350)
        st.plotly_chart(fig_trend, use_container_width=True)

    with col2:
        st.write("**Top 10 wg prognozowanego zapotrzebowania (4 tyg):**")
        top_10 = inv_df.nlargest(10, "forecast_total_4w")
        fig_top = px.bar(
            top_10,
            x="forecast_total_4w",
            y="category",
            orientation="h",
            color="trend",
            color_discrete_map={
                "rosnący 📈": "#2ecc71",
                "stabilny ➡️": "#3498db",
                "spadający 📉": "#e74c3c",
            },
            hover_data=["safety_stock", "reorder_point"],
        )
        fig_top.update_layout(height=400, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_top, use_container_width=True)

    st.divider()

    # ── Tabela z filtrami ──
    st.subheader("📋 Tabela kategorii (filtruj i sortuj)")

    col1, col2 = st.columns([1, 3])
    with col1:
        trend_filter = st.multiselect(
            "Filtruj trend:",
            options=inv_df["trend"].unique(),
            default=inv_df["trend"].unique(),
        )

    filtered = inv_df[inv_df["trend"].isin(trend_filter)]

    display_cols = [
        "category", "forecast_total_4w", "avg_weekly_demand",
        "model_rmse", "safety_stock", "reorder_point",
        "recommended_stock_4w", "trend", "change_pct",
    ]
    display_cols = [c for c in display_cols if c in filtered.columns]

    st.dataframe(
        filtered[display_cols].sort_values("forecast_total_4w", ascending=False),
        use_container_width=True,
        height=400,
    )


# ═══════════════════════════════════════════════════════════════════
# STRONA 2: SZCZEGÓŁY KATEGORII (DRILL-DOWN)
# ═══════════════════════════════════════════════════════════════════

elif page == "🔍 Szczegóły kategorii":
    st.title("🔍 Drill-down: szczegóły kategorii")

    categories = api_get("/categories")
    cat_names = sorted([c["category"] for c in categories])

    selected = st.selectbox("Wybierz kategorię:", cat_names, index=0)

    if selected:
        detail = api_get(f"/categories/{selected}")
        if not detail:
            st.warning("Brak danych dla tej kategorii")
            st.stop()

        summary = detail["summary"]
        weekly = pd.DataFrame(detail["weekly_forecast"])
        trend = detail.get("trend")

        # KPI strip
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Prognoza 4 tyg", f"{summary['forecast_total_4w']:.0f} szt")
        col2.metric("Średnio tygodniowo", f"{summary['avg_weekly_demand']:.1f} szt")
        col3.metric("Safety Stock", f"{summary['safety_stock']:.0f} szt")
        col4.metric("Reorder Point", f"{summary['reorder_point']:.0f} szt")

        col1, col2, col3 = st.columns(3)
        col1.metric("Trend", summary.get("trend", "b/d"))
        if trend:
            col2.metric("Zmiana % (4tyg vs 4tyg)", f"{trend.get('change_pct', 0):+.1f}%")
        col3.metric("RMSE modelu", f"{summary.get('model_rmse', 0):.1f}")

        st.divider()

        # ── Wykres tygodniowej prognozy ──
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("📅 Prognoza tygodniowa")
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=weekly["week_start"],
                y=weekly["forecast"],
                name="Prognoza sprzedaży",
                marker_color="#3498db",
                text=weekly["forecast"].round(0),
                textposition="outside",
            ))
            fig.add_hline(
                y=summary["safety_stock"],
                line_dash="dash",
                line_color="red",
                annotation_text=f"Safety Stock = {summary['safety_stock']:.0f}",
            )
            fig.update_layout(
                xaxis_title="Tydzień",
                yaxis_title="Sztuki",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("💡 Rekomendacja")
            st.success(f"""
            **Rekomendowany stan magazynu na 4 tyg:**

            {summary['recommended_stock_4w']:.0f} szt

            ---
            **Struktura:**
            - Prognoza: {summary['forecast_total_4w']:.0f}
            - Safety Stock: {summary['safety_stock']:.0f}

            **Reorder Point:** {summary['reorder_point']:.0f}

            (zamów więcej gdy stan spadnie poniżej tej wartości)
            """)

        st.divider()

        # ── Produkty w kategorii ──
        st.subheader(f"📦 Produkty w kategorii '{selected}'")
        products = api_get(f"/categories/{selected}/products?limit=100")
        if products:
            col1, col2 = st.columns([1, 3])
            col1.metric("Łącznie produktów", products["total_products"])
            col1.metric("Pokazanych", products["products_shown"])

            with col2:
                st.write("**Lista product_id (top 100):**")
                products_df = pd.DataFrame({"product_id": products["products"]})
                st.dataframe(products_df, use_container_width=True, height=300)


# ═══════════════════════════════════════════════════════════════════
# STRONA 3: UPLOAD CSV
# ═══════════════════════════════════════════════════════════════════

elif page == "📤 Upload magazynu (CSV)":
    st.title("📤 Upload magazynu — rekomendacje per produkt")
    st.caption("Wgraj CSV ze stanami magazynowymi — system zwróci pełne rekomendacje")

    st.info("""
    **Format wymagany:**
    ```
    product_id,current_stock
    abc123def456,150
    def789ghi012,42
    ```
    """)

    uploaded = st.file_uploader("Wybierz plik CSV:", type=["csv"])

    if uploaded:
        # Podgląd
        df_preview = pd.read_csv(uploaded)
        uploaded.seek(0)

        col1, col2 = st.columns([2, 1])
        with col1:
            st.write("**Podgląd wgranego pliku:**")
            st.dataframe(df_preview.head(10), use_container_width=True)
        with col2:
            st.metric("Wierszy", len(df_preview))
            st.metric("Kolumny", ", ".join(df_preview.columns))

        # Wyślij do API
        if st.button("🚀 Generuj rekomendacje", type="primary"):
            with st.spinner("Generowanie rekomendacji..."):
                result = api_post_file(
                    "/recommend",
                    uploaded.getvalue(),
                    uploaded.name,
                )

            if result:
                st.success(f"✓ Wygenerowano rekomendacje dla {result['total_products']} produktów")

                # KPI strip
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Produktów łącznie", result["total_products"])
                col2.metric("Znane kategorie", result["known_products"])
                col3.metric("Nieznane (globalna)", result["unknown_products"])
                col4.metric("Łącznie do zamówienia", f"{result['total_units_to_order']:.0f} szt")

                st.divider()

                # Akcje
                st.subheader("📊 Podsumowanie akcji")
                actions_df = pd.DataFrame(
                    list(result["actions_summary"].items()),
                    columns=["Akcja", "Liczba produktów"],
                )

                action_colors = {
                    "ZAMÓW PILNIE": "#e74c3c",
                    "ZAMÓW": "#e67e22",
                    "OBSERWUJ": "#f1c40f",
                    "OK": "#2ecc71",
                    "NADWYŻKA": "#3498db",
                    "BRAK_DANYCH": "#95a5a6",
                }

                col1, col2 = st.columns([1, 2])
                with col1:
                    fig_actions = px.bar(
                        actions_df,
                        x="Akcja",
                        y="Liczba produktów",
                        color="Akcja",
                        color_discrete_map=action_colors,
                    )
                    fig_actions.update_layout(showlegend=False, height=350)
                    st.plotly_chart(fig_actions, use_container_width=True)

                with col2:
                    # Najpilniejsze produkty
                    recs_df = pd.DataFrame(result["recommendations"])
                    urgent = recs_df[recs_df["urgency"].isin(["high", "medium"])].head(10)
                    if not urgent.empty:
                        st.write("**🚨 Najpilniejsze do zamówienia (top 10):**")
                        st.dataframe(
                            urgent[["product_id", "category", "current_stock",
                                   "units_to_order", "action"]],
                            use_container_width=True,
                            height=350,
                        )
                    else:
                        st.info("Brak pilnych zamówień — wszystkie stany OK")

                st.divider()

                # Pełna tabela
                st.subheader("📋 Pełne rekomendacje (filtruj)")
                recs_df = pd.DataFrame(result["recommendations"])

                col1, col2, col3 = st.columns(3)
                with col1:
                    action_filter = st.multiselect(
                        "Akcja:",
                        options=recs_df["action"].unique(),
                        default=recs_df["action"].unique(),
                    )
                with col2:
                    urgency_filter = st.multiselect(
                        "Pilność:",
                        options=recs_df["urgency"].unique(),
                        default=recs_df["urgency"].unique(),
                    )
                with col3:
                    source_filter = st.multiselect(
                        "Źródło danych:",
                        options=recs_df["data_source"].unique(),
                        default=recs_df["data_source"].unique(),
                    )

                filtered_recs = recs_df[
                    (recs_df["action"].isin(action_filter))
                    & (recs_df["urgency"].isin(urgency_filter))
                    & (recs_df["data_source"].isin(source_filter))
                ]

                st.dataframe(filtered_recs, use_container_width=True, height=500)

                # Download
                csv = filtered_recs.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="💾 Pobierz rekomendacje jako CSV",
                    data=csv,
                    file_name=f"rekomendacje_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                )
