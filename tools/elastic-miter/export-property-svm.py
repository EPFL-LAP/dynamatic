import json
from pprint import pprint

with  open("tools/elastic-miter/out/comp/elastic-miter-config.json") as json_data:
  d = json.load(json_data)

pprint(d)