import argparse
from termcolor import colored
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from internetsearch import internet_search
import pandas as pd

def get_filename():
  parser = argparse.ArgumentParser(description='read arguments')
  parser.add_argument('-f', '--file', type=str, help='input file name')
  
  args = parser.parse_args()
  if args.file:
    return args.file
  else:
    return input("Please enter the filename: ")

def main():
  load_dotenv()
  # GPT_MODEL = "gpt-4o"
  GPT_MODEL = "gpt-3.5-turbo"
  data_points = ["name", "description", "address", "phone", "company_email", "website", "founders"]
  llm = ChatOpenAI(model=GPT_MODEL, temperature=0, streaming=True)

  filename = get_filename()
  try:
    df = pd.read_csv(filename, header = 0)
    output_df = pd.DataFrame(columns=data_points)
    for _, row in df.iterrows():
      output = internet_search(llm, GPT_MODEL, row['Entity'], data_points)
      row_data = {}
      for dp in output["data_points"]:
        row_data[dp["name"]] = dp["value"]
      output_df = pd.concat([output_df, pd.DataFrame([row_data])], ignore_index=True) 
    output_df.to_csv('./output.csv')
  except Exception as e:
      print(colored(f"error while processing file: {e}", "red"))
      return

if __name__ == "__main__":
  main()