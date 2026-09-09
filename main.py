import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="서울 기온 예측기", page_icon="🌡️", layout="centered")

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
CUTOFF_YEAR = 2025          # 수업 기준 기간: 2025년까지
MIN_OBS_PER_YEAR = 300      # 관측일이 300일 미만인 해는 제외


@st.cache_data
def load_data(url: str) -> pd.DataFrame:
    df = pd.read_csv(url, encoding="utf-8")
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df = df.dropna(subset=["날짜", "평균기온"])
    df["연도"] = df["날짜"].dt.year
    return df


@st.cache_data
def build_yearly(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby("연도")["평균기온"]
        .agg(관측일수="count", 연평균기온="mean")
        .reset_index()
    )
    grouped = grouped[
        (grouped["연도"] <= CUTOFF_YEAR) & (grouped["관측일수"] >= MIN_OBS_PER_YEAR)
    ].sort_values("연도").reset_index(drop=True)
    return grouped


st.title("🌡️ 서울 기온 예측기")
st.caption("서울 일별 기온 데이터를 이용해 연도별 평균기온의 추세를 살펴보고, 회귀 직선으로 특정 연도의 예상 기온을 추정합니다.")

try:
    raw_df = load_data(DATA_URL)
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

yearly = build_yearly(raw_df)

if len(yearly) < 2:
    st.error("회귀 직선을 계산하기에 유효한 연도 데이터가 부족합니다.")
    st.stop()

years = yearly["연도"].values.astype(float)
temps = yearly["연평균기온"].values.astype(float)

# 회귀 직선 (최소제곱법)
slope, intercept = np.polyfit(years, temps, 1)
corr = np.corrcoef(years, temps)[0, 1]

n_years = len(yearly)
start_year = int(yearly["연도"].min())
end_year = int(yearly["연도"].max())

# ---- 산점도 + 회귀 직선 ----
line_x = np.array([years.min(), years.max()])
line_y = slope * line_x + intercept

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=years,
        y=temps,
        mode="markers",
        name="연평균기온",
        marker=dict(color="royalblue", size=8),
        hovertemplate="연도: %{x}<br>연평균기온: %{y:.2f}℃<extra></extra>",
    )
)
fig.add_trace(
    go.Scatter(
        x=line_x,
        y=line_y,
        mode="lines",
        name="회귀 직선",
        line=dict(color="crimson", width=2),
    )
)
fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    hovermode="closest",
    height=450,
)
st.plotly_chart(fig, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.metric("상관계수 (r)", f"{corr:.3f}")
with col2:
    st.metric("회귀 직선 기울기", f"{slope:.4f} ℃/년")

st.info(
    f"📌 회귀 직선은 **{n_years}개 연도**의 자료로 만들어졌으며, "
    f"기간은 **{start_year}년 ~ {end_year}년**입니다. "
    f"(기준 기간: {CUTOFF_YEAR}년까지, 연 관측일수 {MIN_OBS_PER_YEAR}일 이상)"
)

st.divider()

# ---- 슬라이더 예측 ----
st.subheader("연도를 선택해 예상 기온을 확인하세요")
selected_year = st.slider("연도 선택", min_value=1900, max_value=2100, value=end_year, step=1)

predicted_temp = slope * selected_year + intercept

if selected_year < start_year or selected_year > end_year:
    st.warning("⚠️ 선택한 연도는 실제 관측 자료 범위를 벗어난 외삽(extrapolation) 예측입니다.")

st.markdown(
    f"""
    <div style="text-align:center; padding: 30px 0;">
        <div style="font-size:22px; color:gray;">{selected_year}년 예상 평균기온</div>
        <div style="font-size:72px; font-weight:bold; color:#d6336c;">{predicted_temp:.2f} ℃</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("원본 연도별 데이터 보기"):
    st.dataframe(yearly, use_container_width=True)
