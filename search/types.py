from pydantic import BaseModel, Field
from typing import Any, Optional


class QueryRequest(BaseModel):
    query: str
    tags: list[str] = Field(default_factory=list)
    k: Optional[int] = None
    providers: list[str] = Field(default_factory=lambda: ["arxiv", "github"])


class ProviderResult(BaseModel):
    id: str
    title: str
    url: str
    abstract: str = ""
    source: str  # "arxiv" | "github" | ...
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    stars: int | None = None
    forks: int | None = None
    citations: int | None = None
    updated_iso: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class WorkItem(BaseModel):
    id: str
    title: str
    url: str
    abstract: str = ""
    source: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    stars: int | None = None
    citations: int | None = None
    updated_iso: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class Cluster(BaseModel):
    id: str
    label: str
    work_ids: list[str]
    centroid: list[float] | None = None
    score: float | None = None


class SearchResponse(BaseModel):
    works: list[WorkItem]
    clusters: list[Cluster]


class ProvidersResponse(BaseModel):
    providers: list[str]
