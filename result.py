from __future__ import annotations

from typing import List

from pydantic import BaseModel


class Result(BaseModel):
    paragraph: str


class Model(BaseModel):
    company_insights: List[Result]
