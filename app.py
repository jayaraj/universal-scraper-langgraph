from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from internetsearch import internet_search
from websitescrap import website_scrap

load_dotenv()
# GPT_MODEL = "gpt-4o"
GPT_MODEL = "gpt-3.5-turbo-1106"

llm = ChatOpenAI(model=GPT_MODEL, temperature=0, streaming=True)

data_points = ["company_name", "company_description", "company_address", "company_phone", "company_email", "company_website"]
entity_name = "Post Qode"
website = "https://postqode.ai"

website_scrap_output = website_scrap(llm, GPT_MODEL, website, entity_name, data_points, [] )
data_points_to_search = [obj["name"] for obj in website_scrap_output["data_points"] if obj ["value"] is None]
internet_search_output = internet_search(llm, GPT_MODEL, entity_name, data_points_to_search)

print("########################################")
print("Requested data points:")
print(data_points)
print("########################################")
print("Data points found:")
for dp in website_scrap_output["data_points"]:
  if dp["value"] is not None:
    print(f"{dp}")

for dp in internet_search_output["data_points"]:
  if dp["value"] is not None:
    print(f"{dp}")
