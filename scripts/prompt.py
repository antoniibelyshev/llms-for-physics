from typing import Optional, Tuple

from torch import Tensor


def gen_prompt_head(
        system_dame: str,
        system_description: Optional[str] = None,
        data_description: Optional[str] = None,
        example_data: Optional[Tuple[Tensor, Tensor]] = None,
):
    prompt = f"Write a torch class Physics(nn.Module) describing {system_dame}.\n {system_description or ''}."
    prompt += 'Return ONLY the code of the class.\n'
    prompt += 'The class initialises learnable parameters, such as length of a pendulum of Jung modulus of material.'
    prompt += 'It also implements forward method which predicts states pass some fixed time given the current state.\n'
    prompt += data_description or ''
    if example_data is not None:
        x, y = example_data
        prompt += f"\nHere is an example of input for the forward method: x={x}, and expected output: y={y}.\n"
    prompt += 'Here are examples of how your answer could look like:\n'

    return prompt
