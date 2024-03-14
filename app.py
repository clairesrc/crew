from flask import Flask
from crewai_tools import SerperDevTool
from result import Model

# from langchain.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain.tools.ddg_search import DuckDuckGoSearchRun

from crewai import Agent, Task, Crew, Process

from datetime import date

def step_callback(step):
    print(step)

stock_analyst = Agent(
 role="Senior Stock Analyst",
 goal= "Report on stocks and analysis with suggestions. ",
 backstory="You're recognized as a major trader in the industry.",
 tools=[
    #   YahooFinanceNewsTool(), 
      DuckDuckGoSearchRun()],
 verbose=True,
 max_iter=10,
 step_callback=step_callback
)

research_analyst = Agent(
    role="Research Analyst",
    goal="Research on the topic and provide a report. ",
    backstory="You're recognized as a major researcher. ",
    tools=[DuckDuckGoSearchRun()],
    verbose=True,
    max_iter=10,
    step_callback=step_callback
    )

writer = Agent(
    role="Writer",
    goal="Write a report that synthesizes the stock trends with today's current events.",
    backstory="You're recognized as a major writer in the industry. You're especially good at synthesizing information and formatting it in a way that is easy to understand and speak out loud.",
    tools=[],
    max_iter=5,
    verbose=True,
    step_callback=step_callback
    )


app = Flask(__name__)

analyze_stock_trends_task = Task(
    name="Analyze Stock Trends",
    description="Look up the stock performance for the company: {company} as of date: {date}. When you have the data you need, list out your findings.  ",
    agent=stock_analyst,
    verbose=True,
    expected_output='A report on the stock performance for the company.'
    )

research_task = Task(
    name="Research",
    description="Research on the company: {company}. ",
    agent=research_analyst,
    verbose=True,
    expected_output='A report on the company\'s recent initiatives, products, or services.'
    )

write_report_task = Task(
    name="Write Report",
    description="Write a report that synthesizes the stock trend with today's current events. Format your answer as plain text with no markup. ",
    agent=writer,
    verbose=True,
    expected_output='A report that synthesizes the stock trend with today\'s current events.'
    )

crew = Crew(
 agents=[stock_analyst, research_analyst, writer],
 tasks=[analyze_stock_trends_task, research_task, write_report_task],
 process=Process.sequential,
 verbose=True,
)

# json
@app.route("/")
def hello_world():
    result = crew.kickoff(inputs={'company': 'Microsoft', 'date': date.today()})
    return result
