# import requests

# url = "http://192.168.49.2:31654/spawn"
# headers = {"Content-Type": "application/json"}
# data = {
#     "image": "timemagic/rl-mcp:proxy-mcp",
#     "ports": [{"container_port": 3000}],
#     "env": {"MCP_LIST": "fetcher-mcp.apps.svc.cluster.local,google-search-mcp.apps.svc.cluster.local"},
#     "expose": "NodePort",
#     "replicas": 3
# }




# response = requests.post(url, json=data, headers=headers)
# print("Status Code:", response.status_code)
# print("Response Body:", response.text)

# import time
# time.sleep(20)  # 等待服务启动

import requests
# uuid = response.json().get("uuid", "default-uuid")
url = f"http://192.168.49.2:31654/deprovision/sandbox-95h3f4b74xl2-a0e86a7a-1270-407b-99d0-5831566ee1d3"

response = requests.delete(url)
print("Status Code:", response.status_code)
print("Response Body:", response.text)