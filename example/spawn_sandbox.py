import requests
import subprocess

# Get the control-plane service URL from minikube
result = subprocess.run(
    ["minikube", "service", "control-plane", "-n", "apps", "--url"],
    capture_output=True,
    text=True,
    check=True
)
CP_URL = result.stdout.strip()


url = f"{CP_URL}/spawn"
headers = {"Content-Type": "application/json"}
data = {
    "image": "sandbox:general-0.1",
    "ports": [{"container_port": 3000}],
    "expose": "ClusterIP",
    "replicas": 1
}

response = requests.post(url, json=data, headers=headers)
print("Status Code:", response.status_code)
print("Response Body:", response.text)


# import requests
# url = f"{CP_URL}/deprovision/{response.json()['']}"

# response = requests.delete(url)
# print("Status Code:", response.status_code)
# print("Response Body:", response.text)

# import requests
# url = f"http://192.168.49.2:30367/deprovision-all"

# response = requests.delete(url)
# print("Status Code:", response.status_code)
# print("Response Body:", response.text)