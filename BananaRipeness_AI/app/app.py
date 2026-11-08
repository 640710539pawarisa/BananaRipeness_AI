
# Banana Ripeness AI Dashboard

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import cv2
import json
import ast
import tempfile

from pathlib import Path
from PIL import Image, ImageOps

# -----------------------------
# 1. Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Banana Ripeness AI",
    page_icon="🍌",
    layout="wide"
)

project_directory = Path(__file__).resolve().parent
results_directory = project_directory / "results"

class_display_names = {
    "freshripe": "Fresh Ripe (กล้วยสุกสด)",
    "freshunripe": "Fresh Unripe (กล้วยดิบสด)",
    "overripe": "Overripe (กล้วยสุกงอม)",
    "ripe": "Ripe (กล้วยสุก)",
    "rotten": "Rotten (กล้วยเน่าเสีย)",
    "unripe": "Unripe (กล้วยดิบ)"
}

# -----------------------------
# 2. Load SVM Prediction System
# -----------------------------

@st.cache_resource
def load_prediction_system():

    model_path = (
        project_directory / "models" / "svm_hsv_best.joblib"
    )

    notebook_path = (
        project_directory / "notebooks" / "03_SVM.ipynb"
    )

    svm_bundle = joblib.load(model_path)

    with open(notebook_path, "r", encoding="utf-8") as file:
        notebook_data = json.load(file)

    config = svm_bundle["feature_config"]

    function_namespace = {
        "np": np,
        "cv2": cv2,
        "Image": Image,
        "ImageOps": ImageOps,
        "image_size": tuple(config["image_size"]),
        "padding_color": tuple(config["padding_color"]),
        "bins_per_channel": config["bins_per_channel"]
    }

    for cell in notebook_data["cells"]:

        if cell["cell_type"] != "code":
            continue

        source_code = "".join(cell["source"])

        try:
            syntax_tree = ast.parse(source_code)
        except SyntaxError:
            continue

        for node in syntax_tree.body:

            if (
                isinstance(node, ast.FunctionDef)
                and node.name == "extract_hsv_features"
            ):

                function_module = ast.Module(
                    body=[node],
                    type_ignores=[]
                )

                exec(
                    compile(
                        function_module,
                        str(notebook_path),
                        "exec"
                    ),
                    function_namespace
                )

                return (
                    svm_bundle,
                    function_namespace["extract_hsv_features"]
                )

    raise RuntimeError("ไม่พบฟังก์ชัน extract_hsv_features")


# -----------------------------
# 3. Sidebar Navigation
# -----------------------------

st.sidebar.title("🍌 Banana Ripeness AI")

st.sidebar.caption(
    "ระบบจำแนกระดับความสุกของกล้วย"
)

selected_page = st.sidebar.radio(
    "เมนูหลัก",
    [
        "🍌 Prediction",
        "📊 Model Comparison",
        "🔍 Model Analysis"
    ]
)

st.sidebar.divider()
st.sidebar.caption("Best Model: SVM")
st.sidebar.caption("Screened Test Accuracy: 95.17%")


# -----------------------------
# PAGE 1: Prediction
# -----------------------------

if selected_page == "🍌 Prediction":

    st.title("🍌 Banana Ripeness Prediction")

    st.write(
        "อัปโหลดภาพกล้วยเพื่อจำแนกระดับความสุกด้วย AI"
    )

    uploaded_image = st.file_uploader(
        "เลือกภาพกล้วย",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image is not None:

        image_column, result_column = st.columns(2)

        with image_column:
            st.image(
                uploaded_image,
                caption="ภาพที่อัปโหลด",
                width=350
            )

        with result_column:

            st.subheader("ผลการทำนาย")

            try:
                svm_bundle, extract_hsv_features = (
                    load_prediction_system()
                )

                with tempfile.TemporaryDirectory() as temp_dir:

                    file_suffix = Path(
                        uploaded_image.name
                    ).suffix

                    image_path = (
                        Path(temp_dir) / f"banana{file_suffix}"
                    )

                    image_path.write_bytes(
                        uploaded_image.getvalue()
                    )

                    image_features = np.asarray(
                        extract_hsv_features(str(image_path)),
                        dtype=np.float32
                    ).reshape(1, -1)

                if image_features.shape[1] != 48:
                    raise ValueError(
                        "จำนวน Features ไม่เท่ากับ 48"
                    )

                scaled_features = svm_bundle[
                    "scaler"
                ].transform(image_features)

                prediction = svm_bundle[
                    "model"
                ].predict(scaled_features)[0]

                predicted_class = svm_bundle[
                    "class_names"
                ][int(prediction)]

                display_name = class_display_names[
                    predicted_class
                ]

                st.success(f"🍌 {display_name}")

                st.caption(
                    "Model: SVM (HSV Histogram + RBF Kernel)"
                )

            except Exception as error:
                st.error("เกิดข้อผิดพลาดในการทำนาย")
                st.exception(error)

    with st.expander("📖 รายละเอียดระดับความสุกทั้ง 6 คลาส"):

        for class_name in class_display_names.values():
            st.write(class_name)

    st.info(
        "ผลทำนายเป็นการประเมินจากภาพด้วยโมเดล AI "
        "อาจคลาดเคลื่อนเมื่อภาพมีแสงหรือพื้นหลังแตกต่างจากข้อมูลฝึก"
    )


# -----------------------------
# PAGE 2: Model Comparison
# -----------------------------

elif selected_page == "📊 Model Comparison":

    st.title("📊 Model Performance Comparison")

    st.write(
        "เปรียบเทียบประสิทธิภาพของโมเดลทั้ง 4 ตัว"
    )

    comparison_path = (
        results_directory / "validation_model_comparison.csv"
    )

    if comparison_path.exists():

        comparison_results = pd.read_csv(
            comparison_path
        )

        comparison_results = comparison_results[
            ["Model", "Accuracy", "Macro_F1"]
        ]

        metric1, metric2, metric3 = st.columns(3)

        metric1.metric(
            "Best Model",
            "SVM"
        )

        metric2.metric(
            "Validation Macro-F1",
            "96.77%"
        )

        metric3.metric(
            "Test Accuracy (SVM)",
            "95.17%"
        )

        st.subheader("ตารางเปรียบเทียบโมเดล")

        st.dataframe(
            comparison_results,
            hide_index=True,
            width="stretch"
        )

        st.subheader("กราฟเปรียบเทียบประสิทธิภาพ")

        st.bar_chart(
            comparison_results,
            x="Model",
            y=["Accuracy", "Macro_F1"],
            stack=False,
            horizontal=True,
            sort=False,
            height=350
        )

        st.caption(
            "Accuracy และ Macro-F1 ในตารางและกราฟ"
            "เป็นผลจาก Screened Validation Set (801 ภาพ)"
        )

        st.success(
            "🏆 SVM มี Validation Macro-F1 สูงที่สุด: 96.77%"
        )

    else:
        st.warning("ไม่พบไฟล์ Validation Comparison")


# -----------------------------
# PAGE 3: Model Analysis
# -----------------------------

elif selected_page == "🔍 Model Analysis":

    st.title("🔍 Model Analysis")

    st.write(
        "วิเคราะห์ Confusion Matrix และ Learning Curves"
    )

    selected_model = st.selectbox(
        "เลือกโมเดล",
        [
            "SVM",
            "EfficientNetV2B0",
            "ResNet50",
            "Custom CNN (Tuned)"
        ]
    )

    analysis_files = {
        "SVM": {
            "confusion": "svm_final_test_confusion_matrix.png",
            "learning": None,
            "dataset": "Screened Test Set (393 ภาพ)"
        },
        "EfficientNetV2B0": {
            "confusion": "efficientnetv2b0_confusion_matrix.png",
            "learning": "efficientnetv2b0_learning_curves.png",
            "dataset": "Screened Validation Set (801 ภาพ)"
        },
        "ResNet50": {
            "confusion": "resnet50_confusion_matrix.png",
            "learning": "resnet50_learning_curves.png",
            "dataset": "Screened Validation Set (801 ภาพ)"
        },
        "Custom CNN (Tuned)": {
            "confusion": "custom_cnn_tuned_confusion_matrix.png",
            "learning": None,
            "dataset": "Screened Validation Set (801 ภาพ)"
        }
    }

    selected_files = analysis_files[selected_model]

    st.info(
        f"ชุดข้อมูล: {selected_files['dataset']}"
    )

    st.subheader("Confusion Matrix")

    confusion_path = (
        results_directory / selected_files["confusion"]
    )

    if confusion_path.exists():
        st.image(str(confusion_path), width=700)
    else:
        st.warning("ไม่พบไฟล์ Confusion Matrix")

    learning_filename = selected_files["learning"]

    if learning_filename is not None:

        st.subheader("Learning Curves")

        learning_path = (
            results_directory / learning_filename
        )

        if learning_path.exists():
            st.image(str(learning_path), width=900)
        else:
            st.warning("ไม่พบไฟล์ Learning Curves")
    else:
        st.caption(
            "ไม่มี Learning Curves สำหรับโมเดลนี้"
        )

# แสดงผลการประเมินหลักทั้ง 4 ค่า
metric1, metric2, metric3, metric4 = st.columns(4)

metric1.metric(
    "Best Model",
    "SVM"
)

metric2.metric(
    "Validation Macro-F1",
    "96.77%"
)

metric3.metric(
    "Test Accuracy",
    "95.17%"
)

metric4.metric(
    "Test Macro-F1",
    "93.09%"
)
