import requests
import subprocess

# Get the control-plane service URL from minikube
result = subprocess.run(
    ["minikube", "service", "control-plane", "-n", "awshive", "--url"],
    capture_output=True,
    text=True,
    check=True
)
CP_URL = result.stdout.strip()

url = f"{CP_URL}/deprovision-all"

response = requests.delete(url)
print("Status Code:", response.status_code)
print("Response Body:", response.text)