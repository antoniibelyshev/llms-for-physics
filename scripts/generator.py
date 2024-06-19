from abc import ABC
from .llm import sample_physics
from .fit import fit
import heapq
from tqdm import trange
from torch import Tensor, stack

from typing import Generator, List, Iterable, Generic, TypeVar, Optional, Callable, Tuple

T = TypeVar('T')


class TopKHeap(Generic[T]):
    def __init__(self, k: int = 16, *, init: Optional[Iterable[T]] = None):
        self.k = k
        self.heap: List[T] = []

        for obj in init:
            self.add(obj)

    def add(self, obj: T):
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, obj)
        else:
            heapq.heappushpop(self.heap, obj)

    def get_top_k(self) -> List[T]:
        return self.heap

    def __repr__(self):
        return f"TopKHeap(k={self.k}, heap={self.heap})"


class PhysicsPredictor(Generator[str], ABC):
    def __init__(
            self,
            sample_data: Callable[[], Tuple[Tensor, Tensor]],
            model: Tensor,
            tokenizer: Tensor,
            prompt_head: str,
            examples: List[str],
            *,
            k: int = 16,
            n_data: int = 1000,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.sample_data = sample_data
        self.n_data = n_data

        self.prompt_head = prompt_head

        self.top_experiments = TopKHeap(init=zip([float('inf') for _ in examples], examples), k=k)

    def __next__(self) -> str:
        return sample_physics(self.top_experiments.get_top_k(), self.model, self.tokenizer, self.prompt_head)

    def train(self, n_iter: int = 100, verbose: bool = True):
        for _ in (trange if verbose else range)(n_iter):
            physics_code = next(self)
            data = self.gen_data()

            model, loss = fit(physics_code, *data)

            self.top_experiments.add((loss, physics_code))

    def gen_data(self, n: Optional[int] = None) -> Tuple[Tensor, Tensor]:
        data = [self.sample_data() for _ in range(n or self.n_data)]
        return stack([d[0] for d in data]), stack([d[1] for d in data])
