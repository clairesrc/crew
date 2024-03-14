from flask import Flask
from crewai_tools import SerperDevTool
from result import Model

from langchain.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain.tools.ddg_search import DuckDuckGoSearchRun

from crewai import Agent, Task, Crew, Process

from datetime import date

def step_callback(step):
    print(step)

stock_analyst = Agent(
 role="Senior Stock Analyst",
 goal= "Report on stocks and analysis with suggestions. ",
 backstory="You're recognized as a major trader in the industry. You work alone without needing help from your team.",
 tools=[
      YahooFinanceNewsTool(), 
      DuckDuckGoSearchRun()],
 verbose=True,
 max_iter=5,
 step_callback=step_callback
)

research_analyst = Agent(
    role="Research Analyst",
    goal="Research on the topic and provide a report. ",
    backstory="You're recognized as a major researcher. ",
    tools=[DuckDuckGoSearchRun()],
    verbose=True,
    can_delegate=True,
    max_iter=10,
    step_callback=step_callback
    )



app = Flask(__name__)

analyze_stock_trends_task = Task(
    name="Analyze Stock Trends",
    description="Look up the stock performance for the company: {company} as of around a week before the date: {date}. When you have the data you need, list out your findings.  ",
    agent=stock_analyst,
    verbose=True,
    expected_output='A report on the stock performance for the company.'
    )

research_task = Task(
    name="Research",
    description="Research on the company: {company}. ",
    agent=research_analyst,
    verbose=True,
    expected_output='A report on the company\'s recent initiatives, products, or services, alongside an analysis of the stock performance of the company as of around a week before {date}.'
    )


crew = Crew(
 agents=[stock_analyst, research_analyst],
 tasks=[analyze_stock_trends_task, research_task],
 process=Process.sequential,
 verbose=True,
)

# json
@app.route("/")
def hello_world():
    result = crew.kickoff(inputs={'company': 'Microsoft', 'date': date.today()})
    return result
