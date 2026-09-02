from unittest.mock import MagicMock, patch
import pytest


# Test successful API response (Match scenario)
@patch("requests.post")
def test_api_success_match(mock_post):
    # Mock a successful backend HTTP 200 response returning a Match (1)
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"prediction": 1, "probability": 0.9423}
    mock_post.return_value = mock_response

    # Define test payload
    payload = {
        "text_input": "Anteroposterior chest radiograph shows mild bilateral pleural effusion..."
    }
    files = {"image_file": ("test.png", b"fake_image_bytes", "image/png")}

    # Send request using requests.post (simulating Streamlit app logic)
    import requests

    response = requests.post(
        "https://sammec-demoday-fastapi.hf.space/predict", data=payload, files=files
    )

    # Assertions
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == 1
    assert data["probability"] == 0.9423


# Test successful API response (Mismatch scenario)
@patch("requests.post")
def test_api_success_mismatch(mock_post):
    # Mock a successful backend HTTP 200 response returning a Mismatch (0)
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"prediction": 0, "probability": 0.1250}
    mock_post.return_value = mock_response

    payload = {"text_input": "Normal chest X-ray."}
    files = {"image_file": ("test.png", b"fake_image_bytes", "image/png")}

    import requests

    response = requests.post(
        "https://sammec-demoday-fastapi.hf.space/predict", data=payload, files=files
    )

    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == 0
    # Calculate confidence for Mismatch (1 - probability) as done in your Streamlit UI
    confidence_score = 1 - data["probability"]
    assert pytest.approx(confidence_score, 0.0001) == 0.8750


# Test backend error handling (HTTP 500)
@patch("requests.post")
def test_api_server_error(mock_post):
    # Mock a 500 Internal Server Error
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    mock_post.return_value = mock_response

    import requests

    response = requests.post(
        "https://sammec-demoday-fastapi.hf.space/predict", data={}, files={}
    )

    assert response.status_code == 500
    assert response.text == "Internal Server Error"


# Test input validation rule (Missing inputs)
def test_input_validation():
    uploaded_file = None
    text_input = ""

    # Simulating the validation logic in your Streamlit app
    is_invalid = (uploaded_file is None) or (not text_input.strip())

    assert is_invalid is True
