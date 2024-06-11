# Universal Webscraper

![Workflow](https://github.com/jayaraj/universal-scraper-langgraph/raw/master/src/img/workflow.png)

## Introduction

Universal Webscraper is a powerful tool designed to extract data points such as company websites, descriptions, founders, emails, addresses, and more, based on a given entity name. The tool accepts input as a CSV file with a column named **Entity** containing the entity names to search and retrieve the requested data points..

## Tools Used

 - **Jina AI** for scrape
 - **Tavily AI** for internet search

You can optionally switch to **FireCrawl** as needed.

## Requirements
  
1.	**Clone this Repository**:
```bash
  git clone https://github.com/jayaraj/universal-scraper-langgraph.git
  cd universal-scraper-langgraph
```
2.	**Install Poetry & Create Environment**:
-	Install Poetry if you haven’t already:
```bash
  pip install poetry
```
-	Install dependencies and activate the virtual environment:
```bash
  poetry install --no-root
  poetry shell
```

3.	**Create a .env File**:
-	Obtain your API keys for OpenAI, Tavily AI, and FireCrawl.
-	Update the .env file with your API keys:
```bash
  OPENAI_API_KEY = "xxxxxxxxxxxxxxxxxxx"
  TAVILY_API_KEY = "xxxxxxxxxxxxxxxxxxx"
  FIRECRAWL_API_KEY="xxxxxxxxxxxxxxxxxxx"
```
4.	**Prepare Input File**:
- Update input.csv with the entity names you want to search 

## Usage

Run the scraper with the following commands:
-	Default run:
```bash
  python app.py
```

- Specify an input file:
```bash
  python app.py -f ./input.csv
```
or
```bash
  python app.py --file ./input.csv
```