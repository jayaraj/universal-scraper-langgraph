from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from internetsearch import internet_search
from websitescrap import website_scrap

load_dotenv()
# GPT_MODEL = "gpt-4o"
GPT_MODEL = "gpt-3.5-turbo"

llm = ChatOpenAI(model=GPT_MODEL, temperature=0, streaming=True)

data_points = ["name", "description", "address", "phone", "company_email", "website", "founders"]
entity_name = "AEOS Labs Ltd India"
# links = ["https://labs.aeoscompany.com/"]

output = internet_search(llm, GPT_MODEL, entity_name, data_points)
# data_points_to_search = [obj["name"] for obj in internet_search_output["data_points"] if obj ["value"] is None]
# website_scrap_output = website_scrap(llm, GPT_MODEL, links, entity_name, internet_search_output, [] )

print("########################################")
print("Data points:")
for dp in output["data_points"]:
  print(f"{dp}")
print("########################################")