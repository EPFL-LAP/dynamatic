import json
from pprint import pprint

with  open("tools/elastic-miter/out/comp/elastic-miter-config.json") as json_data:
    d = json.load(json_data)

pprint(d)


for lhs_buf, rhs_buf in d["input_buffers"]:
    print(lhs_buf + " " + rhs_buf)