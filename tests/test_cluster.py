import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src.cluster import get_book_info, process_book_features, get_recommendations


# ---- Fixtures ----
# fixtures are reusable test data that multiple tests can share

@pytest.fixture
def sample_book_info():
    """a sample book dictionary representing a real book"""
    return {
        "title": "Happy Place",
        "authors": "Emily Henry",
        "subject": "Fiction",
        "categories": "Fiction",
        "page_count": 400,
        "published_date": "2023-04-25",
        "description": "A romance novel about love and family and friendship"
    }

@pytest.fixture
def mock_api_response():
    """a fake API response that mimics what Google Books returns"""
    return {
        "items": [
            {
                "volumeInfo": {
                    "title": "Happy Place",
                    "authors": ["Emily Henry"],
                    "categories": ["Fiction"],
                    "pageCount": 400,
                    "publishedDate": "2023-04-25",
                    "description": "A romance novel about love and family and friendship"
                }
            }
        ]
    }


# ---- Tests for get_book_info ----

def test_get_book_info_returns_dict(mock_api_response):
    """test that get_book_info returns a dictionary"""
    with patch("src.cluster.requests.get") as mock_get:
        mock_get.return_value.json.return_value = mock_api_response
        result = get_book_info("Happy Place")
        assert isinstance(result, dict)

def test_get_book_info_has_required_fields(mock_api_response):
    """test that returned book info has all expected fields"""
    with patch("src.cluster.requests.get") as mock_get:
        mock_get.return_value.json.return_value = mock_api_response
        result = get_book_info("Happy Place")
        assert "title" in result
        assert "authors" in result
        assert "categories" in result
        assert "page_count" in result
        assert "description" in result

def test_get_book_info_no_results():
    """test that get_book_info returns None when no books found"""
    with patch("src.cluster.requests.get") as mock_get:
        mock_get.return_value.json.return_value = {"items": []}
        result = get_book_info("xyznonexistentbookxyz")
        assert result is None


# ---- Tests for process_book_features ----

def test_process_book_features_returns_dataframe(sample_book_info):
    """test that feature vector is a dataframe"""
    result = process_book_features(sample_book_info)
    assert isinstance(result, pd.DataFrame)

def test_process_book_features_correct_shape(sample_book_info):
    """test that feature vector has the correct number of columns"""
    from src.cluster import feature_columns
    result = process_book_features(sample_book_info)
    assert result.shape[1] == len(feature_columns)

def test_process_book_features_no_nan(sample_book_info):
    """test that feature vector contains no NaN values"""
    result = process_book_features(sample_book_info)
    assert result.isna().sum().sum() == 0

def test_process_book_features_values_are_numeric(sample_book_info):
    """test that all feature values are numeric"""
    result = process_book_features(sample_book_info)
    assert all(result.dtypes.apply(lambda x: np.issubdtype(x, np.number)))


# ---- Tests for get_recommendations ----

def test_get_recommendations_returns_correct_count(mock_api_response):
    """test that get_recommendations returns the requested number of books"""
    with patch("src.cluster.requests.get") as mock_get:
        mock_get.return_value.json.return_value = mock_api_response
        book_info, recommendations = get_recommendations("Happy Place", n=5)
        assert len(recommendations) <= 5

def test_get_recommendations_excludes_input_book(mock_api_response):
    """test that the input book is not in the recommendations"""
    with patch("src.cluster.requests.get") as mock_get:
        mock_get.return_value.json.return_value = mock_api_response
        book_info, recommendations = get_recommendations("Happy Place", n=5)
        assert "Happy Place" not in recommendations["title"].values

def test_get_recommendations_has_required_columns(mock_api_response):
    """test that recommendations dataframe has expected columns"""
    with patch("src.cluster.requests.get") as mock_get:
        mock_get.return_value.json.return_value = mock_api_response
        book_info, recommendations = get_recommendations("Happy Place", n=5)
        assert "title" in recommendations.columns
        assert "authors" in recommendations.columns
        assert "subject" in recommendations.columns

def test_get_recommendations_not_found():
    """test that a helpful message is returned when book is not found"""
    with patch("src.cluster.requests.get") as mock_get:
        mock_get.return_value.json.return_value = {"items": []}
        result = get_recommendations("xyznonexistentbookxyz", n=5)
        assert result is not None