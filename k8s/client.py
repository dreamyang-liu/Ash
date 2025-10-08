# import requests

# url = f"http://a8b4ee4606659412ca97fd13254655e1-1667785203.us-west-2.elb.amazonaws.com/spawn"
# headers = {"Content-Type": "application/json"}
# data = {
#     "image": "timemagic/rl-mcp:general-1.6",
#     "ports": [{"container_port": 3000}],
#     "expose": "ClusterIP",
#     "replicas": 1
# }

# for i in range(1):
#     response = requests.post(url, json=data, headers=headers)
#     print("Status Code:", response.status_code)
#     print("Response Body:", response.text)


# import requests
# url = f"http://{IP}:30693/deprovision/sandbox-x0nw9jmbglvu-5010e5d8-7c1d-4b41-b09f-b80dbf591217"

# response = requests.delete(url)
# print("Status Code:", response.status_code)
# print("Response Body:", response.text)

import requests
url = f"http://a8b4ee4606659412ca97fd13254655e1-1667785203.us-west-2.elb.amazonaws.com/deprovision-all"

response = requests.delete(url)
print("Status Code:", response.status_code)
print("Response Body:", response.text)