"""Tests for logging configurations and decorators."""

import logging
from unittest.mock import MagicMock

import pytest

from src.utils.log_wrapper import log_execution
from src.utils.logging_config import setup_logging


def test_setup_logging_dict_config(mocker):
    """Test logging system applies dictConfig correctly."""
    mock_config = mocker.patch("logging.config.dictConfig")
    mock_info = mocker.patch("logging.info")
    
    setup_logging(default_level=logging.DEBUG)
    
    mock_config.assert_called_once()
    config_passed = mock_config.call_args[0][0]
    
    # Assert specific hardcoded settings are established
    assert config_passed["loggers"][""]["level"] == logging.DEBUG
    assert config_passed["loggers"]["httpx"]["level"] == "WARNING"
    assert config_passed["loggers"]["openai"]["level"] == "INFO"
    mock_info.assert_called_once_with("Logging system initialized.")


def test_log_execution_sync_success(mocker):
    """Test wrapper around successful synchronous functions."""
    mock_logger = MagicMock()
    mocker.patch("src.utils.log_wrapper.get_logger", return_value=mock_logger)
    
    @log_execution
    def my_sync_func(data):
        return data + 1
        
    result = my_sync_func(10)
    
    assert result == 11
    # Check start log with truncated args
    assert "🚀 [START]" in mock_logger.info.call_args_list[0][0][0]
    assert "my_sync_func" in mock_logger.info.call_args_list[0][0][0]
    assert "Args: (10)" in mock_logger.info.call_args_list[0][0][0]
    # Check success log
    assert "✅ [SUCCESS]" in mock_logger.info.call_args_list[1][0][0]


def test_log_execution_sync_failure(mocker):
    """Test wrapper correctly logging and reraising on synchronous exceptions."""
    mock_logger = MagicMock()
    mocker.patch("src.utils.log_wrapper.get_logger", return_value=mock_logger)
    
    @log_execution
    def my_sync_fail():
        raise ValueError("Critical crash!")
        
    with pytest.raises(ValueError, match="Critical crash!"):
        my_sync_fail()
        
    error_str = mock_logger.error.call_args[0][0]
    assert "❌ [ERROR]" in error_str
    assert "my_sync_fail" in error_str
    assert "Fail: Critical crash!" in error_str
