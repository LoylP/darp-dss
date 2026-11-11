"""Giao diện Streamlit tiếng Việt cho hệ thống DARP."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from api_client import APIClient, APIError

COLORS = {"MV": "#9aa5b1", "WM": "#e0a458", "DARP": "#2f6db5"}
PAPER_ETA = {"f1": 3.54, "f2": 2.63, "f3": 1.10, "f4": 1.27, "f5": 1.48, "f6": 3.39}
MODEL_COLORS = {
    **COLORS,
    "DARP_T3": COLORS["DARP"],
    "DARP_T5": COLORS["DARP"],
    "DARP_C": COLORS["DARP"],
    **{f"-f{index}": COLORS["DARP"] for index in range(1, 7)},
}
RATING_NAMES = {1: "Sell", 2: "Under", 3: "Hold", 4: "Buy", 5: "Strong Buy"}
API_URL = os.getenv("API_URL", "http://localhost:8000")
client = APIClient(API_URL)

st.set_page_config(page_title="DARP – Hỗ trợ quyết định", page_icon="📊", layout="wide")
st.title("DARP – Hệ thống hỗ trợ quyết định")
st.caption("Decision aggregation with reliability propagation")


@st.cache_data(ttl=300)
def get_meta(base_url: str) -> dict[str, Any]:
    """Lưu đệm danh sách lựa chọn do API cung cấp."""
    return APIClient(base_url).meta()


@st.cache_data(ttl=300)
def get_results(base_url: str) -> dict[str, Any]:
    """Lưu đệm kết quả phân tích do API cung cấp."""
    return APIClient(base_url).results()


def _show_api_error(error: APIError) -> None:
    st.error(str(error))


try:
    meta = get_meta(API_URL)
    results = get_results(API_URL)
except APIError as error:
    st.error(str(error))
    st.stop()

tabs = st.tabs(
    ["Kết quả tái hiện", "Ra quyết định", "Ví dụ minh họa", "Chạy thực nghiệm"]
)

with tabs[0]:
    st.subheader("Kết quả tái hiện")
    overall = results.get("ACC_overall", {})
    intervals = results.get("ACC_CI95", {})
    columns = st.columns(3)
    for column, model in zip(columns, ("MV", "WM", "DARP")):
        low, high = intervals.get(model, [float("nan"), float("nan")])
        column.metric(
            f"ACC {model}",
            f"{overall.get(model, float('nan')):.4f}",
            help=f"Khoảng tin cậy 95%: [{low:.4f}, {high:.4f}]",
        )

    year_values = results.get("ACC_by_year", {})
    if year_values:
        year_frame = pd.DataFrame(year_values).rename_axis("Năm").reset_index()
        year_frame["Năm"] = year_frame["Năm"].astype(str)
        year_frame = year_frame.melt(
            id_vars="Năm", var_name="Mô hình", value_name="ACC"
        )
        chart = px.line(
            year_frame,
            x="Năm",
            y="ACC",
            color="Mô hình",
            markers=True,
            color_discrete_map=COLORS,
            title="Fig. 3–4 (tái hiện) – ACC tổng hợp theo năm",
        )
        chart.update_xaxes(type="category")
        st.plotly_chart(chart, use_container_width=True)
        st.caption("So sánh ACC của MV, WM và DARP theo năm.")

    for key, title, x_title in (
        ("acc_by_sector", "Fig. 5 (tái hiện) – ACC theo ngành", "Ngành"),
        ("acc_by_nrecs", "Fig. 6 (tái hiện) – ACC theo số khuyến nghị", "Số khuyến nghị"),
    ):
        data = results.get(key, {})
        if data:
            frame = pd.DataFrame(data)
            frame.index.name = x_title
            frame = frame.reset_index()
            frame[x_title] = frame[x_title].astype(str)
            frame = frame.melt(
                id_vars=[x_title, "n"],
                value_vars=["MV", "WM", "DARP"],
                var_name="Mô hình",
                value_name="ACC",
            )
            chart = px.bar(
                frame,
                x=x_title,
                y="ACC",
                color="Mô hình",
                barmode="group",
                color_discrete_map=COLORS,
                hover_data={"n": True},
                title=title,
            )
            chart.update_xaxes(type="category")
            st.plotly_chart(chart, use_container_width=True)
            st.caption("So sánh: các giá trị được tính từ artifacts tái lập.")

    topk = results.get("fig7_topk", {})
    if topk:
        frame = pd.DataFrame({"Mô hình": list(topk), "ACC": list(topk.values())})
        st.plotly_chart(
            px.bar(
                frame,
                x="Mô hình",
                y="ACC",
                color="Mô hình",
                color_discrete_map=MODEL_COLORS,
                title="Fig. 7 (tái hiện) – Bỏ phiếu top-k",
            ),
            use_container_width=True,
        )
        st.caption("So sánh: DARP, DARP_T3 và DARP_T5 từ cùng tập sự kiện.")

    ablation = results.get("fig8_ablation", {})
    if ablation:
        frame = pd.DataFrame({"Mô hình": list(ablation), "ACC": list(ablation.values())})
        st.plotly_chart(
            px.bar(
                frame,
                x="Mô hình",
                y="ACC",
                color="Mô hình",
                color_discrete_map=MODEL_COLORS,
                title="Fig. 8 (tái hiện) – Ablation",
            ),
            use_container_width=True,
        )
        st.caption("So sánh: mỗi hàng loại bỏ một thành phần hoặc thay thế trọng số liên quan.")

    sensitivity = results.get("fig9_sensitivity", [])
    if sensitivity:
        frame = pd.DataFrame(sensitivity)
        charts = st.columns(3)
        for column, parameter in zip(charts, ("TP", "OB", "PD")):
            subset = frame.loc[frame.param == parameter]
            if not subset.empty:
                subset = subset.melt(
                    id_vars=["param", "value"],
                    value_vars=["MV", "WM", "DARP"],
                    var_name="Mô hình",
                    value_name="ACC",
                )
                chart = px.bar(
                    subset,
                    x="value",
                    y="ACC",
                    color="Mô hình",
                    barmode="group",
                    color_discrete_map=COLORS,
                    category_orders={
                        "value": (
                            [7, 14, 30]
                            if parameter == "OB"
                            else [1, 2, 3]
                            if parameter == "TP"
                            else [1, 3, 5]
                        )
                    },
                    title=f"Fig. 9 (tái hiện) – {parameter}",
                )
                chart.update_xaxes(type="category")
                column.plotly_chart(chart, use_container_width=True)
        st.caption("Độ nhạy theo các thiết lập TP, OB và PD.")

    eta = results.get("eta_mean", {})
    if eta:
        st.subheader("Độ liên quan trung bình η")
        st.dataframe(
            pd.DataFrame(
                {"Package": eta, "Bài báo": PAPER_ETA}
            ).rename_axis("Đặc trưng"),
            use_container_width=True,
        )
    reliability_year = results.get("reliability_by_year", {})
    if reliability_year:
        frame = pd.DataFrame(
            {"Năm": list(reliability_year), "mean": list(reliability_year.values())}
        )
        st.plotly_chart(
            px.bar(
                frame,
                x="Năm",
                y="mean",
                title="Fig. 10 (tái hiện) – Độ tin cậy theo năm",
            ).update_xaxes(type="category"),
            use_container_width=True,
        )
        st.caption("So sánh: trung bình c theo năm từ các nút kiểm tra.")

    reliability_sector = results.get("reliability_by_sector", {})
    sector_means = reliability_sector.get("mean", {})
    if sector_means:
        frame = pd.DataFrame(
            {"Ngành": list(sector_means), "mean": list(sector_means.values())}
        )
        st.plotly_chart(
            px.bar(
                frame,
                x="mean",
                y="Ngành",
                orientation="h",
                title="Fig. 11 (tái hiện) – Độ tin cậy theo ngành",
            ),
            use_container_width=True,
        )
        st.caption("So sánh: trung bình c theo ngành từ các nút kiểm tra.")

    heatmap = results.get("reliability_heatmap", {})
    if heatmap:
        frame = pd.DataFrame(heatmap).T
        frame.index = frame.index.astype(str)
        frame.columns = frame.columns.astype(str)
        st.plotly_chart(
            px.imshow(
                frame,
                labels={"x": "Năm", "y": "Ngành", "color": "Trung bình c"},
                title="Fig. 12 (tái hiện) – Năm × ngành",
                aspect="auto",
                color_continuous_scale="RdBu",
                color_continuous_midpoint=0,
                origin="lower",
            ),
            use_container_width=True,
        )
        st.caption("So sánh: trung bình c giao giữa ngành và năm.")

    quintiles = results.get("reliability_quintiles", {})
    if quintiles:
        st.subheader("Độ tin cậy theo ngũ phân vị")
        st.dataframe(pd.DataFrame(quintiles), use_container_width=True)

with tabs[1]:
    st.subheader("Ra quyết định")
    months = meta.get("months", [])
    if not months:
        st.info("API chưa có tháng nào để hiển thị.")
    else:
        month = st.selectbox("Tháng đánh giá", months, key="event_month")
        only_disagree = st.checkbox("Chỉ hiện sự kiện DARP khác MV")
        try:
            event_rows = client.events(month, only_disagree, limit=500)
            if not event_rows:
                st.info("Không có sự kiện phù hợp.")
            else:
                events_frame = pd.DataFrame(event_rows)
                st.dataframe(events_frame, use_container_width=True, hide_index=True)
                event_id = st.selectbox(
                    "Sự kiện",
                    events_frame.event_id.tolist(),
                    format_func=lambda value: (
                        f"#{value} – {events_frame.loc[events_frame.event_id == value, 'ticker'].iloc[0]}"
                    ),
                )
                detail = client.event_detail(int(event_id))
                st.markdown(
                    f"**Nhãn thật:** {RATING_NAMES.get(detail['y'], detail['y'])}"
                )
                st.markdown("**Độ liên quan η**")
                st.dataframe(
                    pd.DataFrame(
                        {
                            "Package": detail["eta"],
                            "Bài báo": PAPER_ETA,
                        }
                    ).rename_axis("Đặc trưng"),
                    use_container_width=True,
                )
                node_frame = pd.DataFrame(detail["nodes"])
                node_frame["n_train"] = node_frame["n_train"].map(
                    lambda count: "chưa có lịch sử" if count == 0 else str(count)
                ).astype(str)
                st.dataframe(
                    node_frame[
                        [
                            "analyst",
                            "broker",
                            "rating_name",
                            "c",
                            "wm_weight",
                            "n_train",
                            "true_q",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
                decisions = detail["decisions"]
                result_cols = st.columns(3)
                for column, model in zip(result_cols, ("MV", "WM", "DARP")):
                    correct = detail["correct"][model]
                    column.metric(
                        model, RATING_NAMES.get(decisions[model], decisions[model])
                    )
                    column.markdown("✅" if correct else "❌")
                with st.expander("Vì sao? – láng giềng ảnh hưởng"):
                    for row in detail["nodes"]:
                        st.markdown(f"**{row['analyst']}**")
                        influence_frame = pd.DataFrame(row["influences"])
                        if influence_frame.empty:
                            st.info("Không có láng giềng train ảnh hưởng.")
                        else:
                            st.dataframe(
                                influence_frame[
                                    [
                                        "rec_id",
                                        "analyst",
                                        "ticker",
                                        "date",
                                        "x",
                                        "y",
                                        "c",
                                        "contribution",
                                    ]
                                ],
                                use_container_width=True,
                                hide_index=True,
                            )
        except APIError as error:
            _show_api_error(error)

    st.markdown("#### Sự kiện mới")
    if meta.get("months") and meta.get("tickers") and meta.get("analysts"):
        with st.form("decide_form"):
            new_month = st.selectbox("Tháng", meta["months"], key="new_month")
            ticker_options = [row["ticker"] for row in meta["tickers"]]
            new_ticker = st.selectbox("Ticker", ticker_options)
            analyst_options = [row["analyst"] for row in meta["analysts"]]
            default_recs = pd.DataFrame(
                {"analyst": analyst_options[:2], "rating": [3, 4]}
            )
            new_recs = st.data_editor(
                default_recs,
                num_rows="dynamic",
                column_config={
                    "analyst": st.column_config.SelectboxColumn(
                        "Chuyên gia", options=analyst_options, required=True
                    ),
                    "rating": st.column_config.NumberColumn(
                        "Nhãn (1–5)", min_value=1, max_value=5, step=1, required=True
                    ),
                },
                use_container_width=True,
                key="new_recommendations",
            )
            submitted = st.form_submit_button("Đánh giá")
        if submitted:
            recommendations = [
                {"analyst": row.analyst, "rating": int(row.rating)}
                for row in new_recs.itertuples(index=False)
                if pd.notna(row.analyst) and pd.notna(row.rating)
            ]
            try:
                result = client.decide(
                    {"month": new_month, "ticker": new_ticker, "recommendations": recommendations}
                )
                new_node_frame = pd.DataFrame(result["nodes"])
                new_node_frame["n_train"] = new_node_frame["n_train"].map(
                    lambda count: "chưa có lịch sử" if count == 0 else str(count)
                ).astype(str)
                st.dataframe(
                    new_node_frame[
                        ["analyst", "rating_name", "c", "wm_weight", "n_train"]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
                st.markdown("**Độ liên quan η**")
                st.dataframe(
                    pd.DataFrame(
                        {
                            "Package": result["eta"],
                            "Bài báo": PAPER_ETA,
                        }
                    ).rename_axis("Đặc trưng"),
                    use_container_width=True,
                )
                for row in result["nodes"]:
                    st.markdown(f"**Láng giềng ảnh hưởng – {row['analyst']}**")
                    influences = pd.DataFrame(row["influences"])
                    if influences.empty:
                        st.info("Không có láng giềng train ảnh hưởng.")
                    else:
                        st.dataframe(
                            influences[
                                [
                                    "rec_id",
                                    "analyst",
                                    "ticker",
                                    "date",
                                    "x",
                                    "y",
                                    "c",
                                    "contribution",
                                ]
                            ],
                            use_container_width=True,
                            hide_index=True,
                        )
                st.json(result["decisions"])
            except APIError as error:
                _show_api_error(error)

with tabs[2]:
    st.subheader("Ví dụ minh họa")
    initial_rows = pd.DataFrame(
        [
            {"ev": 0, "analyst": "a1", "broker": "X", "sic": 7372, "x": 5, "y": 5},
            {"ev": 0, "analyst": "a4", "broker": "Z", "sic": 7372, "x": 3, "y": 5},
            {"ev": 1, "analyst": "a2", "broker": "Y", "sic": 7371, "x": 2, "y": 4},
            {"ev": 1, "analyst": "a4", "broker": "Z", "sic": 7371, "x": 4, "y": 4},
            {"ev": 2, "analyst": "a1", "broker": "X", "sic": 7373, "x": 4, "y": None},
            {"ev": 2, "analyst": "a2", "broker": "Y", "sic": 7373, "x": 2, "y": None},
            {"ev": 2, "analyst": "a3", "broker": "Y", "sic": 7373, "x": 2, "y": None},
        ]
    )
    toy_rows = st.data_editor(initial_rows, num_rows="dynamic", use_container_width=True)
    if st.button("Chạy DARP"):
        rows = []
        for row in toy_rows.to_dict(orient="records"):
            rows.append(
                {
                    **row,
                    "ev": int(row["ev"]),
                    "sic": int(row["sic"]),
                    "x": int(row["x"]),
                    "y": None if pd.isna(row["y"]) else int(row["y"]),
                }
            )
        try:
            output = client.toy({"rows": rows})
            st.dataframe(pd.DataFrame(output["nodes"]), use_container_width=True)
            st.success(
                f"MV chọn {RATING_NAMES[output['MV']]}; DARP chọn {RATING_NAMES[output['DARP']]}."
            )
        except APIError as error:
            _show_api_error(error)

with tabs[3]:
    st.subheader("Chạy thực nghiệm")
    st.warning("Một lượt có thể mất 1–3 phút.")
    with st.form("experiment_form"):
        tp = st.slider("Training Period (tháng)", 1, 3, 1)
        ob = st.select_slider("Observational Horizon (ngày)", [7, 14, 30], value=14)
        pd_horizon = st.select_slider("Predictive Horizon (ngày làm việc)", [1, 3, 5], value=1)
        n_months = st.slider("Số tháng gần nhất", 3, 12, 12)
        run = st.form_submit_button("Chạy")
    if run:
        with st.spinner("Đang tính toán..."):
            try:
                output = client.experiment(
                    {"TP": tp, "OB": ob, "PD": pd_horizon, "n_months": n_months}
                )
                frame = pd.DataFrame(
                    {"Mô hình": list(output["ACC"]), "ACC": list(output["ACC"].values())}
                )
                st.plotly_chart(
                    px.bar(frame, x="Mô hình", y="ACC", color="Mô hình", color_discrete_map=COLORS),
                    use_container_width=True,
                )
                ci_frame = pd.DataFrame(output["ACC_CI95"], index=["Cận dưới", "Cận trên"]).T
                st.dataframe(ci_frame, use_container_width=True)
                st.caption(
                    f"CI bootstrap 95%; n={output['n_test_events']} sự kiện. "
                    f"Thời gian chạy: {output['seconds']:.1f} giây."
                )
                if output["n_test_events"] < 30:
                    st.warning(
                        "Cỡ mẫu dưới 30 sự kiện; khoảng tin cậy có thể không ổn định."
                    )
            except APIError as error:
                _show_api_error(error)
