from torch import Tensor
from torch.optim import Optimizer, LBFGS
from torch import nn
import torch

from typing import Type, Optional, Dict, Any, Callable, Tuple
from tqdm import trange


def fit(
        model_definition: str,
        x: Tensor,
        y: Tensor,
        *,
        optim: Type[Optimizer] = LBFGS,
        optim_defaults: Optional[Dict[str, Any]] = None,
        loss_fn: Callable[[Tensor, Tensor], Tensor] = nn.functional.mse_loss,
        n_iter: int = 100,
        verbose: bool = True,
) -> Tuple[nn.Module, float]:
    namespace: Dict[str, Any] = {'torch': torch, 'nn': nn}
    exec(model_definition, namespace)
    model = namespace['Physics']()
    assert isinstance(model, nn.Module)

    optimizer = optim(model.parameters(), **optim_defaults or {})

    closure = create_closure(model, x, y, loss_fn, optimizer)

    for it in (trange if verbose else range)(n_iter):
        optimizer.step(closure)

    return model, loss_fn(model(x), y).item()


def create_closure(
        model: nn.Module,
        x: Tensor,
        y: Tensor,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        optimizer: Optimizer,
) -> Callable[[], float]:
    def closure() -> float:
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        return loss.item()

    return closure
