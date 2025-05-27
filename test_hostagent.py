#!/usr/bin/env python3
import requests
import json
import time
import sys

base_url = "http://localhost:8080"

def make_request(endpoint, payload):
    """Make a request to the host agent and return the response"""
    url = f"{base_url}/{endpoint}"
    
    print(f"Making request to {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        if endpoint == "get_output":
            # For get_output, we use GET request with query parameters
            response = requests.get(
                url, 
                params={"trajectory_id": payload["trajectory_id"], "id": payload["id"]}
            )
        else:
            # For other endpoints, we use POST request with JSON payload
            response = requests.post(url, json=payload)
        
        response.raise_for_status()
        print(f"Response status: {response.status_code}")
        
        if response.content:
            try:
                result = response.json()
                print(f"Response: {json.dumps(result, indent=2)}")
                return result
            except ValueError:
                print(f"Response (text): {response.text}")
                return response.text
        return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None


def test_start_sandbox(trajectory_id="test-trajectory-1"):
    """Test starting a sandbox container"""
    payload = {
        "id": "start-001",
        "trajectory": trajectory_id,
        "request_type": 2,  # REQUEST_TYPE_START_SANDBOX
        "start_sandbox_input": {
            "image_id": "ubuntu:latest",
            "user": "root",
            "working_dir": "/testbed",
            "network_disabled": False,
            "shell_path": "/bin/bash"
        }
    }
    
    return make_request("start_sandbox", payload)


def test_run_command(cmd, request_id="cmd-001", trajectory_id="test-trajectory-1"):
    """Test running a command in the sandbox"""
    payload = {
        "id": request_id,
        "trajectory": trajectory_id,
        "request_type": 0,  # REQUEST_TYPE_RUN_COMMAND
        "run_command_input": {
            "command": cmd,
            "working_dir": "/testbed",
            "timeout_in_seconds": 3,
            "network_disabled": False,
            "shell_path": "/bin/bash",
            "is_interactive": True
        }
    }
    
    return make_request("run_command", payload)


def test_get_output(request_id="cmd-001", trajectory_id="test-trajectory-1"):
    """Test getting output from a command execution"""
    payload = {
        "id": request_id,
        "trajectory_id": trajectory_id
    }
    
    return make_request("get_output", payload)


def test_shutdown_sandbox(trajectory_id="test-trajectory-1"):
    """Test shutting down a sandbox container"""
    payload = {
        "id": "shutdown-001",
        "trajectory": trajectory_id,
        "request_type": 3,  # REQUEST_TYPE_SHUTDOWN_SANDBOX
    }
    
    return make_request("shutdown_sandbox", payload)


def run_test_sequence():
    """Run a sequence of tests to demonstrate the host agent functionality"""
    trajectory_id = "test-trajectory-1"
    
    # Step 1: Start the sandbox
    print("\n=== Starting sandbox ===")
    test_start_sandbox()
    time.sleep(2)  # Wait for container to start
    
    # Step 2: Run some commands
    print("\n=== Running command: ls -la ===")
    test_run_command("ls -la", "cmd-001", trajectory_id)
    time.sleep(1)
    
    print("\n=== Getting output ===")
    test_get_output("cmd-001", trajectory_id)
    time.sleep(1)
    
    print("\n=== Running command: echo 'Hello World' > test.txt ===")
    test_run_command("echo 'Hello World' > test.txt", "cmd-002", trajectory_id)
    time.sleep(1)
    
    print("\n=== Running command: cat test.txt ===")
    test_run_command("cat test.txt", "cmd-003", trajectory_id)
    time.sleep(1)
    
    print("\n=== Getting output ===")
    test_get_output("cmd-003", trajectory_id)
    time.sleep(1)
    
    # Step 3: Install a package
    print("\n=== Running command: apt-get update -y && apt-get install -y curl ===")
    test_run_command("apt-get update -y && apt-get install -y curl", "cmd-004", trajectory_id)
    time.sleep(10)  # This will take longer
    
    print("\n=== Running command: curl --version ===")
    test_run_command("curl --version", "cmd-005", trajectory_id)
    time.sleep(1)
    
    print("\n=== Getting output ===")
    test_get_output("cmd-005", trajectory_id)
    time.sleep(1)
    
    # Step 4: Shutdown the sandbox
    print("\n=== Shutting down sandbox ===")
    test_shutdown_sandbox(trajectory_id)


def test_nginx_git():
    """Run a sequence of tests to demonstrate the host agent functionality"""
    trajectory_id = "test-trajectory-nginx"
    
    # Step 1: Start the sandbox
    print("\n=== Starting sandbox ===")
    test_start_sandbox(trajectory_id)
    time.sleep(2)  # Wait for container to start
    
    # Step 3: Install a package
    print("\n=== Running command: apt-get update -y && apt-get install -y git ===")
    test_run_command("apt-get update -y && apt-get install -y git", "cmd-004", trajectory_id)
    time.sleep(40)  # This will take longer

    test_get_output("cmd-004", trajectory_id)
    
    print("\n=== Running command: curl --version ===")
    test_run_command("git clone https://github.com/nginx/nginx.git && cd nginx && rm README.md", "cmd-005", trajectory_id)
    time.sleep(10)

    print("\n=== Running command: curl --version ===")
    test_run_command("git --no-pager diff", "cmd-005", trajectory_id)
    time.sleep(1)

    test_get_output("cmd-005", trajectory_id)

    print("\n=== Running command: curl --version ===")
    test_run_command("ls -al --color", "cmd-005", trajectory_id)
    time.sleep(1)

    test_get_output("cmd-005", trajectory_id)

    print("\n=== Getting output ===")
    test_get_output("cmd-005", trajectory_id)
    time.sleep(1)
    
    # Step 4: Shutdown the sandbox
    print("\n=== Shutting down sandbox ===")
    test_shutdown_sandbox(trajectory_id)

def main():
    if len(sys.argv) > 1:
        action = sys.argv[1]
        if action == "start":
            test_start_sandbox()
        elif action == "run" and len(sys.argv) > 2:
            request_id = f"cmd-{int(time.time())}"
            test_run_command(sys.argv[2], request_id)
            time.sleep(0.5)
            test_get_output(request_id)
        elif action == "output" and len(sys.argv) > 2:
            test_get_output(sys.argv[2])
        elif action == "shutdown":
            test_shutdown_sandbox()
        else:
            print("Invalid action")
            print("Usage: python test_hostagent.py [start|run \"command\"|output request_id|shutdown]")
    else:
        print("Running test sequence...")
        test_nginx_git()


if __name__ == "__main__":
    main()