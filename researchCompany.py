from langchain.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain.tools.ddg_search import DuckDuckGoSearchRun
from langchain.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from result import Model

from langchain.utilities.tavily_search import TavilySearchAPIWrapper
from langchain.tools.tavily_search import TavilySearchResults
from langchain_community.tools.polygon.aggregates import PolygonAggregates
from langchain_community.tools.polygon.financials import PolygonFinancials
from langchain_community.tools.polygon.last_quote import PolygonLastQuote
from langchain_community.tools.polygon.ticker_news import PolygonTickerNews
from langchain_community.utilities.polygon import PolygonAPIWrapper

from crewai import Agent, Task, Crew, Process

from datetime import date




wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
tavily_tool = TavilySearchResults(api_wrapper=TavilySearchAPIWrapper())
polygon_wrapper = PolygonAPIWrapper()
last_quote = PolygonLastQuote(api_wrapper=polygon_wrapper)
aggregates = PolygonAggregates(api_wrapper=polygon_wrapper)
financials = PolygonFinancials(api_wrapper=polygon_wrapper)

def step_callback(step):
    print(step)


def researchCompany(company):
    stock_analyst = Agent(
        role="Senior Analyst",
        goal="Report on a given company's current news and analysis. ",
        backstory="You're recognized as a major trader in the industry. ",
        tools=[
            
                YahooFinanceNewsTool(), 
                DuckDuckGoSearchRun(), 
                wikipedia, 
                tavily_tool,
                # last_quote,
                # financials,
                # aggregates
                ],
        verbose=True,
        max_rpm=10,
        max_iter=15,
        step_callback=step_callback,
    )

    analyze_stock_trends_task = Task(
        name="Analyze Stock Trends",
        description="Look up the recent news related to the company: {company} leading up to the day before today's date: {date}. Do not make up any information, look everything up with the tools you have available. Once you have the data you need, summarize your findings using exact figures as necessary. Wikipedia should not be used to retrieve current stock information. Do not add comments to your Action Input. Only one Action can be invoked at once, and the Action name should always exactly match one of the options given with nothing else added to it. Action inputs should be formatted as unescaped JSON. ",
        agent=stock_analyst,
        verbose=True,
        expected_output="A several-paragraph report on the company's recent initiatives, products, or services, alongside an analysis of the stock performance of the company leading up to today's date {date}. As a summary, try to find relationships between the stock performance and the company's recent activities.",
        output_json=Model
    )

    crew = Crew(
        agents=[stock_analyst],
        tasks=[analyze_stock_trends_task],
        process=Process.sequential,
        verbose=True,
    )
    result = crew.kickoff(inputs={"company": company, "date": date.today()})
    return result
