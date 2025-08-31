#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MCP server that integrates with Sandbox Fusion code interpreter
"""

import json
import logging
import os
import sys
from typing import Dict, List, Optional, Any, Union

from fastmcp import FastMCP
from sandbox_fusion import (
    run_code, RunCodeRequest,
    run_jupyter, RunJupyterRequest,
    set_endpoint
)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # DEBUG level for more detailed logs
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("mcp_server.log"), logging.StreamHandler()]
)
logger = logging.getLogger("mcp-sandbox-fusion")

import subprocess

try:
    process = subprocess.Popen(["sh", "-lc", "bash /root/sandbox/scripts/run.sh"], stderr=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True)
except Exception as e:
    logger.error(f"Failed to launch subprocess: {str(e)}")
    sys.exit(1)

# Set Sandbox Fusion API endpoint
endpoint = os.environ.get("SANDBOX_FUSION_ENDPOINT", "http://localhost:8080")
set_endpoint(endpoint)
logger.info(f"Sandbox Fusion endpoint set to: {endpoint}")

# Create MCP server instance
mcp_server = FastMCP("Sandbox Fusion Code Interpreter")

# Supported languages
SUPPORTED_LANGUAGES = [
    "python", "javascript", "typescript", "bash", "r", "julia", 
    "golang", "java", "cpp", "rust", "php"
]

@mcp_server.resource("sandbox://languages")
def list_languages() -> str:
    """Return list of supported programming languages"""
    try:
        return json.dumps(SUPPORTED_LANGUAGES, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Failed to get language list: {str(e)}")
        return json.dumps({"error": str(e)})

@mcp_server.resource("sandbox://endpoint")
def get_endpoint() -> str:
    """Return current Sandbox Fusion endpoint info"""
    try:
        return json.dumps({"endpoint": endpoint}, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Failed to get endpoint info: {str(e)}")
        return json.dumps({"error": str(e)})

@mcp_server.tool()
def execute_code(code: str, language: str = "python", timeout: Optional[int] = None) -> str:
    """
    Execute a code snippet

    Parameters:
    - code: The code to execute
    - language: Programming language
    - timeout: Timeout in seconds
    """
    try:
        logger.info(f"Executing code: language={language}, code length={len(code)}")
        
        if language not in SUPPORTED_LANGUAGES:
            return json.dumps({"error": f"Unsupported language: {language}. Supported: {', '.join(SUPPORTED_LANGUAGES)}"})
        
        request = RunCodeRequest(code=code, language=language)
        response = run_code(request, client_timeout=timeout)
        
        return json.dumps(response.dict(), ensure_ascii=False)
    except Exception as e:
        logger.error(f"Code execution failed: {str(e)}")
        return json.dumps({"error": str(e)})

@mcp_server.tool()
def execute_jupyter(notebook_content: str, timeout: Optional[int] = None) -> str:
    """
    Execute a Jupyter notebook

    Parameters:
    - notebook_content: Jupyter notebook content (JSON format)
    - timeout: Timeout in seconds
    """
    try:
        logger.info(f"Executing Jupyter notebook: content length={len(notebook_content)}")
        
        # Validate notebook JSON
        try:
            notebook_json = json.loads(notebook_content)
        except json.JSONDecodeError:
            return json.dumps({"error": "Invalid Jupyter notebook content. Must be valid JSON."})
        
        request = RunJupyterRequest(notebook=notebook_json)
        response = run_jupyter(request, client_timeout=timeout)
        
        return json.dumps(response.dict(), ensure_ascii=False)
    except Exception as e:
        logger.error(f"Jupyter execution failed: {str(e)}")
        return json.dumps({"error": str(e)})

@mcp_server.prompt()
def code_execution_prompt(language: str = "python", description: str = "") -> str:
    """
    Create a code execution prompt template

    Parameters:
    - language: Programming language
    - description: Task description
    """
    template = {
        "description": f"Execute {language} code" + (f": {description}" if description else ""),
        "language": language,
        "task": description or f"Implement and run code in {language}"
    }
    
    return json.dumps(template, ensure_ascii=False)

@mcp_server.prompt()
def jupyter_execution_prompt(description: str = "") -> str:
    """
    Create a Jupyter notebook execution prompt template

    Parameters:
    - description: Task description
    """
    template = {
        "description": "Execute Jupyter notebook" + (f": {description}" if description else ""),
        "task": description or "Create and execute a Jupyter notebook"
    }
    
    return json.dumps(template, ensure_ascii=False)

if __name__ == "__main__":
    logger.info("Starting MCP server using stdio transport")
    try:
        # Redirect stderr to a log file to avoid interfering with stdio transport
        sys.stderr = open("mcp_stderr.log", "w")
        
        # Run MCP server
        mcp_server.run(transport="streamable-http", host="0.0.0.0", port=3000)
    except Exception as e:
        logger.error(f"Server runtime error: {str(e)}", exc_info=True)
