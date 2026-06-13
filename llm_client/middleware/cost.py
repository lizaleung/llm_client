import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterator, List

from rich.console import Console
from rich.table import Table

from ..base import BaseLLMClient
from ..types import LLMResponse, Message


@dataclass
class _CallRecord:
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float


class CostTracker(BaseLLMClient):
    def __init__(self, client: BaseLLMClient):
        self._client = client
        self._records: List[_CallRecord] = []

    @property
    def model(self) -> str:
        return self._client.model

    def complete(self, messages: List[Message], **kwargs) -> LLMResponse:
        response = self._client.complete(messages, **kwargs)
        self._records.append(
            _CallRecord(
                model=response.model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cost_usd=response.usage.cost_usd,
                latency_ms=response.latency_ms,
            )
        )
        return response

    def stream(self, messages: List[Message], **kwargs) -> Iterator[str]:
        result = self._client.stream(messages, **kwargs)
        start = time.perf_counter()

        def _gen():
            yield from result
            usage = getattr(result, "usage", None)
            if usage is not None:
                self._records.append(
                    _CallRecord(
                        model=getattr(result, "model", None) or self.model,
                        input_tokens=usage.input_tokens,
                        output_tokens=usage.output_tokens,
                        cost_usd=usage.cost_usd,
                        latency_ms=(time.perf_counter() - start) * 1000,
                    )
                )

        return _gen()

    @property
    def total_cost_usd(self) -> float:
        return sum(r.cost_usd for r in self._records)

    @property
    def total_input_tokens(self) -> int:
        return sum(r.input_tokens for r in self._records)

    @property
    def total_output_tokens(self) -> int:
        return sum(r.output_tokens for r in self._records)

    @property
    def call_count(self) -> int:
        return len(self._records)

    def reset(self) -> None:
        self._records.clear()

    def print_summary(self) -> None:
        console = Console()

        if not self._records:
            console.print("[yellow]No API calls recorded.[/yellow]")
            return

        table = Table(title="LLM Cost Summary", show_footer=True)
        table.add_column("Model", footer="[bold]TOTAL[/bold]")
        table.add_column("Calls", justify="right", footer=str(self.call_count))
        table.add_column("Input Tokens", justify="right", footer=str(self.total_input_tokens))
        table.add_column("Output Tokens", justify="right", footer=str(self.total_output_tokens))
        table.add_column("Cost (USD)", justify="right", footer=f"[bold]${self.total_cost_usd:.6f}[/bold]")
        table.add_column("Avg Latency (ms)", justify="right")

        groups: dict[str, list[_CallRecord]] = defaultdict(list)
        for r in self._records:
            groups[r.model].append(r)

        for model_name, records in groups.items():
            avg_latency = sum(r.latency_ms for r in records) / len(records)
            table.add_row(
                model_name,
                str(len(records)),
                str(sum(r.input_tokens for r in records)),
                str(sum(r.output_tokens for r in records)),
                f"${sum(r.cost_usd for r in records):.6f}",
                f"{avg_latency:.1f}",
            )

        console.print(table)
