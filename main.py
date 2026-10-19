
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# --------------------------------------------------
# 1. 페이지 설정
# --------------------------------------------------
st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 영화 유형 나누기")
st.caption(
    "영화의 흥행 특성을 분석하고 비슷한 영화끼리 묶습니다."
)

DATA_URL = (
    "https://raw.githubusercontent.com/it-teacher-go/"
    "data-share-/main/kobis_movies.csv"
)

FEATURE_LABELS = {
    "log_first_scrn": "스크린 수 (log₁₀)",
    "log_total_audi": "누적 관객 수 (log₁₀)",
    "days_in_top10": "10위권 일수",
    "longrun_index": "롱런 지수"
}

CLUSTER_NAMES = [
    "㉮", "㉯", "㉰", "㉱", "㉲", "㉳", "㉴"
]

FEATURE_COLUMNS = list(FEATURE_LABELS.keys())


# --------------------------------------------------
# 2. 데이터 불러오기 및 전처리
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    required_columns = [
        "movieCd", "movieNm", "openDt", "genre", "nation",
        "first_scrn", "first_show", "first_date", "peak",
        "first_week_audi", "total_audi", "days_in_top10"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "필수 열이 없습니다: " + ", ".join(missing_columns)
        )

    numeric_columns = [
        "first_scrn",
        "first_week_audi",
        "total_audi",
        "days_in_top10"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col]
            .astype("string")
            .str.replace(",", "", regex=False),
            errors="coerce"
        )

    total_count = len(df)

    # 로그 변환과 롱런 지수 계산에 필요한 값이 있는 영화만 유지
    valid = (
        df["first_scrn"].notna()
        & df["total_audi"].notna()
        & df["first_week_audi"].notna()
        & df["days_in_top10"].notna()
        & (df["first_scrn"] > 0)
        & (df["total_audi"] > 0)
        & (df["first_week_audi"] > 0)
    )

    for col in numeric_columns:
        valid &= np.isfinite(df[col])

    df = df.loc[valid].copy()

    # 파생 속성
    df["log_first_scrn"] = np.log10(df["first_scrn"])
    df["log_total_audi"] = np.log10(df["total_audi"])

    df["longrun_index"] = (
        df["total_audi"] / df["first_week_audi"]
    ).clip(upper=20)

    # 분석에 필요한 속성에 결측 또는 무한대가 없는지 재확인
    feature_frame = df[FEATURE_COLUMNS].replace(
        [np.inf, -np.inf], np.nan
    )

    usable = feature_frame.notna().all(axis=1)
    df = df.loc[usable].copy()

    return df, total_count


try:
    df, total_count = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.info("인터넷 연결과 데이터 URL을 확인해 주세요.")
    st.stop()


# --------------------------------------------------
# 3. 군집화 속성 및 묶음 수 선택
# --------------------------------------------------
st.subheader("1. 군집화 설정")

selected_features = st.multiselect(
    "군집화에 사용할 속성 (2개 이상)",
    options=FEATURE_COLUMNS,
    default=FEATURE_COLUMNS,
    format_func=lambda x: FEATURE_LABELS[x],
    key="cluster_features"
)

cluster_count = st.selectbox(
    "묶음 수 (K)",
    options=list(range(2, 8)),
    index=1,
    help="2개부터 7개까지 선택할 수 있습니다.",
    key="cluster_count"
)

st.markdown(
    f"**전체 영화: {total_count:,}편　|　"
    f"군집화 대상: "
    f"{len(df):,}편**"
)

if len(selected_features) < 2:
    st.warning("군집화 속성을 2개 이상 선택해 주세요.")
    st.stop()

if len(df) < 3:
    st.error("군집화를 하려면 유효한 영화 데이터가 최소 3편 필요합니다.")
    st.stop()

if len(df) < cluster_count:
    st.error(
        f"선택한 묶음 수가 {cluster_count}개이지만 "
        f"분석 가능한 영화는 {len(df)}편입니다."
    )
    st.stop()


# --------------------------------------------------
# 4. 선택한 속성 표준화
# --------------------------------------------------
X = df[selected_features].to_numpy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# --------------------------------------------------
# 5. 선택한 묶음 수로 K-means 군집화
# --------------------------------------------------
kmeans = KMeans(
    n_clusters=cluster_count,
    random_state=42,
    n_init=10
)

df = df.copy()
df["_cluster_raw"] = kmeans.fit_predict(X_scaled)

# 누적 관객 평균이 높은 군집부터 ㉮, ㉯, ㉰ ... 지정
cluster_order = (
    df.groupby("_cluster_raw", observed=True)["total_audi"]
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)

cluster_name_map = {
    raw_cluster: CLUSTER_NAMES[i]
    for i, raw_cluster in enumerate(cluster_order)
}

df["군집"] = df["_cluster_raw"].map(cluster_name_map)

df["군집"] = pd.Categorical(
    df["군집"],
    categories=CLUSTER_NAMES[:cluster_count],
    ordered=True
)

active_cluster_names = CLUSTER_NAMES[:cluster_count]


# --------------------------------------------------
# 6. 엘보 분석: K=1~7의 군집 내 제곱거리 합
# --------------------------------------------------
st.divider()
st.subheader("2. 묶음 수 결정하기: 엘보 그래프")

st.write(
    "세로축은 각 영화와 자신이 속한 군집 중심 사이의 "
    "거리 제곱을 모두 더한 값입니다. "
    "묶음 수가 증가하면 이 값은 대체로 감소합니다."
)

max_k = min(7, len(df))
inertia_values = []

for k in range(1, max_k + 1):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    model.fit(X_scaled)

    inertia_values.append({
        "묶음 수": k,
        "군집 내 제곱거리 합": model.inertia_
    })

inertia_df = pd.DataFrame(inertia_values)

fig_elbow = go.Figure()

fig_elbow.add_trace(
    go.Scatter(
        x=inertia_df["묶음 수"],
        y=inertia_df["군집 내 제곱거리 합"],
        mode="lines+markers",
        name="제곱거리 합",
        line=dict(width=3),
        marker=dict(size=8),
        hovertemplate=(
            "묶음 수: %{x}<br>"
            "제곱거리 합: %{y:,.3f}"
            "<extra></extra>"
        )
    )
)

# 현재 선택한 묶음 수에 세로선 표시
fig_elbow.add_vline(
    x=cluster_count,
    line_width=2,
    line_dash="dash",
    line_color="red",
    annotation_text=f"현재 K={cluster_count}",
    annotation_position="top"
)

fig_elbow.update_layout(
    xaxis_title="묶음 수 (K)",
    yaxis_title="군집 내 제곱거리 합",
    xaxis=dict(
        tickmode="linear",
        dtick=1
    ),
    hovermode="x unified",
    showlegend=False
)

st.plotly_chart(fig_elbow, use_container_width=True)

# 바로 앞 묶음 수와 비교한 감소량
inertia_table = inertia_df.copy()

inertia_table["직전 값 대비 감소량"] = (
    inertia_table["군집 내 제곱거리 합"].shift(1)
    - inertia_table["군집 내 제곱거리 합"]
)

# 첫 행은 비교할 직전 값이 없으므로 빈칸
inertia_table.loc[0, "직전 값 대비 감소량"] = np.nan

inertia_table["군집 내 제곱거리 합"] = (
    inertia_table["군집 내 제곱거리 합"].map(
        lambda x: f"{x:,.3f}"
    )
)

inertia_table["직전 값 대비 감소량"] = (
    inertia_table["직전 값 대비 감소량"].map(
        lambda x: "" if pd.isna(x) else f"{x:,.3f}"
    )
)

st.dataframe(
    inertia_table,
    use_container_width=True,
    hide_index=True
)

if max_k < 7:
    st.warning(
        f"분석 가능한 영화가 {len(df)}편이므로 "
        f"엘보 그래프는 K=1부터 K={max_k}까지만 계산했습니다."
    )


# --------------------------------------------------
# 7. 실루엣 점수
# --------------------------------------------------
st.divider()
st.subheader("3. 현재 묶음의 실루엣 점수")

current_labels = df["_cluster_raw"].to_numpy()
unique_labels = np.unique(current_labels)

if (
    len(unique_labels) >= 2
    and len(unique_labels) < len(df)
):
    silhouette = silhouette_score(
        X_scaled,
        current_labels
    )

    st.markdown(
        f"**현재 묶음 수 {cluster_count}의 실루엣 점수: "
        f"{silhouette:.4f}**"
    )

    st.caption(
        "실루엣 점수는 -1에서 1 사이입니다. "
        "1에 가까울수록 같은 군집 안의 영화는 가깝고 "
        "다른 군집의 영화와는 멀리 분리되어 있음을 뜻합니다."
    )
else:
    st.warning(
        "현재 군집 결과에서는 실루엣 점수를 계산할 수 없습니다."
    )


# --------------------------------------------------
# 8. 2차원 산점도
# --------------------------------------------------
st.divider()
st.subheader("4. 2차원 산점도")

col_x, col_y = st.columns(2)

with col_x:
    x_feature = st.selectbox(
        "가로축 속성",
        options=selected_features,
        index=0,
        format_func=lambda x: FEATURE_LABELS[x],
        key="scatter_x"
    )

with col_y:
    y_feature = st.selectbox(
        "세로축 속성",
        options=selected_features,
        index=1,
        format_func=lambda x: FEATURE_LABELS[x],
        key="scatter_y"
    )

fig2 = px.scatter(
    df,
    x=x_feature,
    y=y_feature,
    color="군집",
    category_orders={"군집": active_cluster_names},
    hover_name="movieNm",
    hover_data={
        "군집": True,
        x_feature: ":.3f",
        y_feature: ":.3f",
        "movieNm": False
    },
    labels={
        x_feature: FEATURE_LABELS[x_feature],
        y_feature: FEATURE_LABELS[y_feature],
        "군집": "영화 유형"
    },
    title=f"영화 유형별 분포 (K={cluster_count})"
)

fig2.update_traces(
    marker=dict(size=7, opacity=0.75)
)

fig2.update_layout(
    legend_title_text="영화 유형",
    hoverlabel=dict(namelength=-1)
)

st.plotly_chart(fig2, use_container_width=True)


# --------------------------------------------------
# 9. 3차원 산점도
# --------------------------------------------------
st.divider()
st.subheader("5. 3차원 산점도")

if len(selected_features) < 3:
    st.info(
        "3차원 산점도를 표시하려면 "
        "군집화 속성을 3개 이상 선택해 주세요."
    )
else:
    axis_cols = st.columns(3)

    with axis_cols[0]:
        x3 = st.selectbox(
            "X축",
            options=selected_features,
            index=0,
            format_func=lambda x: FEATURE_LABELS[x],
            key="x3"
        )

    with axis_cols[1]:
        y3 = st.selectbox(
            "Y축",
            options=selected_features,
            index=1,
            format_func=lambda x: FEATURE_LABELS[x],
            key="y3"
        )

    with axis_cols[2]:
        z3 = st.selectbox(
            "Z축",
            options=selected_features,
            index=2,
            format_func=lambda x: FEATURE_LABELS[x],
            key="z3"
        )

    fig3 = px.scatter_3d(
        df,
        x=x3,
        y=y3,
        z=z3,
        color="군집",
        category_orders={"군집": active_cluster_names},
        hover_name="movieNm",
        hover_data={
            "군집": True,
            x3: ":.3f",
            y3: ":.3f",
            z3: ":.3f",
            "movieNm": False
        },
        labels={
            x3: FEATURE_LABELS[x3],
            y3: FEATURE_LABELS[y3],
            z3: FEATURE_LABELS[z3],
            "군집": "영화 유형"
        },
        title=f"영화 유형별 3차원 분포 (K={cluster_count})"
    )

    fig3.update_traces(
        marker=dict(size=3, opacity=0.8)
    )

    fig3.update_layout(
        legend_title_text="영화 유형",
        hoverlabel=dict(namelength=-1),
        scene=dict(
            xaxis_title=FEATURE_LABELS[x3],
            yaxis_title=FEATURE_LABELS[y3],
            zaxis_title=FEATURE_LABELS[z3]
        )
    )

    st.plotly_chart(fig3, use_container_width=True)


# --------------------------------------------------
# 10. 군집별 편수 및 원래 단위의 평균
# --------------------------------------------------
st.divider()
st.subheader("6. 군집별 특성 비교")

summary_rows = []

for name in active_cluster_names:
    group = df[df["군집"] == name]

    if group.empty:
        continue

    summary_rows.append({
        "영화 유형": name,
        "편수": len(group),
        "스크린 수 평균": group["first_scrn"].mean(),
        "누적 관객 수 평균": group["total_audi"].mean(),
        "10위권 일수 평균": group["days_in_top10"].mean(),
        "롱런 지수 평균": group["longrun_index"].mean()
    })

summary = pd.DataFrame(summary_rows)

if not summary.empty:
    display_summary = summary.copy()

    display_summary["스크린 수 평균"] = (
        display_summary["스크린 수 평균"].round(1)
    )
    display_summary["누적 관객 수 평균"] = (
        display_summary["누적 관객 수 평균"].round(1)
    )
    display_summary["10위권 일수 평균"] = (
        display_summary["10위권 일수 평균"].round(2)
    )
    display_summary["롱런 지수 평균"] = (
        display_summary["롱런 지수 평균"].round(2)
    )

    st.dataframe(
        display_summary,
        use_container_width=True,
        hide_index=True
    )


# --------------------------------------------------
# 11. 군집별 누적 관객 상위 5편
# --------------------------------------------------
st.divider()
st.subheader("7. 군집별 누적 관객 상위 영화")

for name in active_cluster_names:
    group = (
        df[df["군집"] == name]
        .sort_values(
            ["total_audi", "movieNm"],
            ascending=[False, True]
        )
        .head(5)
    )

    if group.empty:
        st.markdown(f"### {name} 유형")
        st.caption("해당 유형에 속한 영화가 없습니다.")
        continue

    st.markdown(
        f"### {name} 유형 · 상위 {len(group)}편"
    )

    top_movies = group[
        ["movieNm", "total_audi"]
    ].copy()

    top_movies.columns = [
        "영화 제목",
        "누적 관객 수"
    ]

    top_movies["누적 관객 수"] = (
        top_movies["누적 관객 수"]
        .map(lambda n: f"{n:,.0f}명")
    )

    st.dataframe(
        top_movies,
        use_container_width=True,
        hide_index=True
    )


# --------------------------------------------------
# 12. 안내
# --------------------------------------------------
st.divider()

st.caption(
    "분석 방법: 선택한 속성 표준화 → K-means 군집화. "
    "난수 시드 42, n_init=10을 사용합니다. "
    "군집 기호는 누적 관객 평균이 높은 순서로 지정합니다. "
    "엘보 그래프는 같은 표준화 데이터에 대해 K=1~7을 "
    "계산하며, 실루엣 점수는 현재 선택한 K로 계산합니다."
)
