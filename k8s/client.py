import requests

url = "http://192.168.49.2:30367/spawn"
headers = {"Content-Type": "application/json"}
data = {
    "image": "timemagic/rl-mcp:general",
    "ports": [{"container_port": 3000}],
    "env": {"MCP_HUB_ADDR": "proxy-mcp.apps.svc.cluster.local:3000/mcp"},
    "expose": "ClusterIP",
    "replicas": 1
}

response = requests.post(url, json=data, headers=headers)
print("Status Code:", response.status_code)
print("Response Body:", response.text)


import requests
url = f"http://192.168.49.2:30367/deprovision/sandbox-x0nw9jmbglvu-5010e5d8-7c1d-4b41-b09f-b80dbf591217"

response = requests.delete(url)
print("Status Code:", response.status_code)
print("Response Body:", response.text)

import requests
url = f"http://192.168.49.2:30367/deprovision-all"

response = requests.delete(url)
print("Status Code:", response.status_code)
print("Response Body:", response.text)