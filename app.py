from flask import Flask
from flask import request
from datetime import date
from crewai import crew

from researchCompany import researchCompany

app = Flask(__name__)


# json
@app.route("/top")
def top():
    return "top5"


@app.route("/research")
def research():
    return researchCompany(request.args.get('company'))
