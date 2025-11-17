from abc import ABC, abstractmethod
from typing import List, Optional
from search.types import ProviderResult


class Provider(ABC):
    name: str

    @abstractmethod
    async def search(
        self,
        query: str,
        limit: int,
        *,
        earliest_date: Optional[str] = None,  # "YYYY-MM-DD"
        min_popularity: Optional[int] = None
    ) -> List[ProviderResult]:
        ...
