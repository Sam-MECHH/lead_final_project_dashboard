import streamlit as st
import requests
from PIL import Image

# Set up page configurations
st.set_page_config(page_title="BioVil-T Multi-Modal Checker", layout="wide")

# App Titles and Descriptions
st.title("Chest X-Ray Vs Report Coherence Checker")
st.write(
    "Upload a chest X-ray image and enter the corresponding clinical report text to analyze alignment."
)

st.write("---")

# Layout: Split into two large columns for inputs
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Patient Radio")
    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image...", type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        # Display the image cleanly in the column
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Patient Image", use_container_width=True)

with col2:
    st.subheader("2. Radiology Report Text")
    text_input = st.text_area(
        "Enter clinical report",
        placeholder="e.g., Anteroposterior chest radiograph shows mild bilateral pleural effusion...",
        height=250,
    )

st.write("---")

# Action Area: Button to trigger prediction
st.subheader("3. Execution & Results")
if st.button("Launch tne check", type="primary"):
    if uploaded_file is None or not text_input.strip():
        st.error(
            "⚠️ Please provide BOTH a chest X-ray image and clinical text before analyzing."
        )
    else:
        with st.spinner("In progress ..."):
            try:
                # 🔴 CRITICAL: Point this to your live Hugging Face Space API endpoint!
                # Note the '/predict' suffix
                api_url = "https://sammec-demoday-fastapi.hf.space/predict"

                # Prepare the multi-part form data payload
                # Reset file pointer to start just in case
                uploaded_file.seek(0)
                files = {
                    "image_file": (
                        uploaded_file.name,
                        uploaded_file.read(),
                        uploaded_file.type,
                    )
                }
                data = {"text_input": text_input}

                # Send the request to your FastAPI container
                response = requests.post(api_url, data=data, files=files)

                if response.status_code == 200:
                    result = response.json()
                    prediction = result.get("prediction")  # Expecting 1 or 0
                    probability = result.get("probability", 0.0)

                    # Layout for the visual match indicator circles
                    res_col1, res_col2 = st.columns([1, 4])

                    with res_col1:
                        # Dynamic Circle Indicator (Green for Match/1, Red for Mismatch/0)
                        if prediction == 1:
                            st.markdown("### 🟢 **MATCH**")
                        else:
                            st.markdown("### 🔴 **MISMATCH**")

                    with res_col2:
                        # Display the raw score
                        st.metric(
                            label="Prediction Confidence Score",
                            value=f"{probability if prediction == 1 else (1 - probability):.4f}",
                        )

                else:
                    st.error(
                        f"Backend API Error (Status {response.status_code}): {response.text}"
                    )

            except Exception as e:
                st.error(f"Could not connect to the remote server. Details: {e}")
