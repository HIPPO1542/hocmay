from io import BytesIO
import os
from pathlib import Path
from time import perf_counter

import altair as alt
import pandas as pd
import streamlit as st

from src.compare_models import load_comparison
from src.config import MODEL_NAMES, PROJECT_ROOT, Paths
from src.model_artifacts import model_signature
from src.predict import load_predictor
from src.preprocessing import read_image
from src.utils import ProjectError

PATHS = Paths(Path(os.environ.get("FRUIT_VEG_WORKSPACE", PROJECT_ROOT)).resolve())
UPLOAD_BYTES = 15 * 1024 ** 2
METRIC_LABELS = {
    "model": "Mô hình",
    "accuracy": "Accuracy",
    "macro_precision": "Macro Precision",
    "macro_recall": "Macro Recall",
    "macro_f1": "Macro F1",
    "train_seconds": "Huấn luyện (s)",
    "inference_ms": "Suy luận (ms/ảnh)",
    "model_size_mib": "Dung lượng (MiB)",
}


@st.cache_resource(show_spinner=False)
def cached_predictor(root: str, key: str, signature: tuple[int, int]):
    return load_predictor(Paths(Path(root)), key)


def recognition_page() -> None:
    st.header("Nhận diện từ ảnh")
    model_key = st.selectbox("Mô hình", list(MODEL_NAMES), format_func=MODEL_NAMES.get)
    available = PATHS.model_file(model_key).is_file() and PATHS.metadata_file(model_key).is_file()
    if not available:
        st.warning(f"Chưa có mô hình {MODEL_NAMES[model_key]} đã huấn luyện.")
    upload = st.file_uploader("Tải ảnh rau củ quả", type=["jpg", "jpeg", "png"],
                              help="Chọn tệp JPEG hoặc PNG tối đa 15 MB.")
    if upload is None:
        return
    if upload.size > UPLOAD_BYTES:
        st.error("Tệp vượt quá 15 MB.")
        return

    image_bytes = upload.getvalue()
    try:
        image = read_image(BytesIO(image_bytes), upload=True)
    except ProjectError as error:
        st.error(str(error))
        return

    preview, result = st.columns([1, 1], gap="large")
    with preview:
        image.thumbnail((1000, 1000))
        st.image(image, caption="Ảnh đã tải lên", width="stretch")
    with result:
        if not st.button("Nhận diện", type="primary", disabled=not available):
            return
        try:
            with st.spinner("Đang nhận diện ảnh…"):
                predictor = cached_predictor(str(PATHS.root), model_key, model_signature(PATHS, model_key))
                started = perf_counter()
                prediction = predictor.predict(read_image(BytesIO(image_bytes), upload=True))
                elapsed_ms = (perf_counter() - started) * 1000
        except (ProjectError, OSError, ValueError, RuntimeError, KeyError) as error:
            st.error(f"Không thể nhận diện ảnh. {error}")
            return

        st.metric("Kết quả dự đoán", prediction.class_name)
        if prediction.probability is None:
            st.info("Mô hình SVM này chưa được hiệu chỉnh xác suất.")
        else:
            st.metric("Xác suất của nhãn dự đoán", f"{prediction.probability:.1%}")
            st.subheader("Top 3 theo xác suất")
            ranking = pd.DataFrame(prediction.top, columns=["Lớp", "Xác suất"])
            st.dataframe(ranking, hide_index=True, width="stretch",
                         column_config={"Xác suất": st.column_config.NumberColumn(format="percent")})
            chart = alt.Chart(ranking).mark_bar(color="#236c52", cornerRadiusEnd=4).encode(
                x=alt.X("Xác suất:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
                y=alt.Y("Lớp:N", sort="-x", title=None),
                tooltip=["Lớp:N", alt.Tooltip("Xác suất:Q", format=".1%")],
            )
            st.altair_chart(chart, width="stretch")
            if model_key == "svm" and prediction.top[0][0] != prediction.class_name:
                st.caption("Nhãn SVC và thứ hạng theo xác suất hiệu chỉnh có thể khác nhau.")
        st.caption(f"Thời gian xử lý {elapsed_ms:.1f} ms · Đọc ảnh, tiền xử lý và dự đoán")


def comparison_page() -> None:
    st.header("So sánh mô hình")
    try:
        comparison = load_comparison(PATHS)
    except (ProjectError, OSError, ValueError, KeyError) as error:
        st.info(str(error))
        return

    st.caption(f"{comparison['test_count']} ảnh test hợp lệ · Suy luận CPU, batch 1")
    rows = pd.DataFrame(comparison["rows"]).rename(columns=METRIC_LABELS)
    formats = {
        label: st.column_config.NumberColumn(format="%.4f" if index < 5 else "%.2f")
        for index, label in enumerate(METRIC_LABELS.values()) if index > 0
    }
    st.dataframe(rows, hide_index=True, width="stretch", column_config=formats)
    st.download_button("Tải bảng CSV", data=(PATHS.metrics / "model_comparison.csv").read_bytes(),
                       file_name="model_comparison.csv", mime="text/csv")
    overview = PATHS.figures / "model_comparison.png"
    if overview.is_file():
        st.image(str(overview), width="stretch")
    st.markdown(comparison["recommendation"])

    for key in comparison["model_keys"]:
        title = MODEL_NAMES[key]
        with st.expander(f"Chi tiết {title}"):
            st.write(comparison["tradeoffs"][title])
            pairs_path = PATHS.metrics / f"{key}_confusion_pairs.csv"
            if pairs_path.is_file():
                pairs = pd.read_csv(pairs_path).head(10)
                if not pairs.empty:
                    st.dataframe(
                        pairs.rename(columns={"true_label": "Nhãn thực", "predicted_label": "Dự đoán", "count": "Số ảnh"}),
                        hide_index=True,
                    )
            for stem in ("confusion_matrix", "mistakes"):
                figure = PATHS.figures / f"{key}_{stem}.png"
                if figure.is_file():
                    st.image(str(figure), width="stretch")


def main() -> None:
    st.set_page_config(page_title="Fruit & Vegetable Recognition AI", page_icon="🥬", layout="wide")
    st.title("Fruit & Vegetable Recognition AI")
    if (PATHS.root / ".smoke_run.json").is_file():
        st.warning("Đang mở dữ liệu smoke test tổng hợp. Chỉ dùng để kiểm tra kỹ thuật.")
    st.sidebar.header("Điều hướng")
    page = st.sidebar.radio("Trang", ["Nhận diện", "So sánh mô hình"], label_visibility="collapsed")
    if page == "Nhận diện":
        recognition_page()
    else:
        comparison_page()


if __name__ == "__main__":
    main()
