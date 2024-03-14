from langchain.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain.tools.ddg_search import DuckDuckGoSearchRun
from langchain.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

from langchain.utilities.tavily_search import TavilySearchAPIWrapper
from langchain.tools.tavily_search import TavilySearchResults

from crewai import Agent, Task, Crew, Process

from datetime import date


wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
tavily_tool = TavilySearchResults(api_wrapper=TavilySearchAPIWrapper())


def step_callback(step):
    print(step)


def researchCompany(company):
    stock_analyst = Agent(
        role="Senior Stock Analyst",
        goal="Report on stocks and analysis with suggestions. ",
        backstory="You're recognized as a major trader in the industry. ",
        tools=[YahooFinanceNewsTool()],
        verbose=True,
        max_rpm=10,
        max_iter=10,
        step_callback=step_callback,
    )

    research_analyst = Agent(
        role="Research Analyst",
        goal="Research on the topic and provide a report. ",
        backstory="You're recognized as a major researcher. ",
        tools=[DuckDuckGoSearchRun(), wikipedia, tavily_tool],
        verbose=True,
        max_rpm=10,
        can_delegate=True,
        max_iter=10,
        step_callback=step_callback,
    )

    analyze_stock_trends_task = Task(
        name="Analyze Stock Trends",
        description="Look up the stock performance for the company: {company} as of around a week before the date: {date}. When you have the data you need, list out your findings. Only one Action can be invoked at once, and the Action name should always exactly match one of the options given with nothing else added to it. Action inputs should be formatted as unescaped JSON. ",
        agent=stock_analyst,
        verbose=True,
        expected_output="A report on the stock performance for the company.",
    )

    research_task = Task(
        name="Research",
        description="Research on the company: {company} using search engine data as of the date: {date}. Only one Action can be invoked at once, and the Action name should always exactly match one of the options given with nothing else added to it. Action inputs should be formatted as unescaped JSON.",
        agent=research_analyst,
        verbose=True,
        expected_output="A report on the company's recent initiatives, products, or services, alongside an analysis of the stock performance of the company as of around a week before {date}.",
    )

    crew = Crew(
        agents=[stock_analyst, research_analyst],
        tasks=[analyze_stock_trends_task, research_task],
        process=Process.sequential,
        verbose=True,
    )
    result = crew.kickoff(inputs={"company": company, "date": date.today()})
    return result
