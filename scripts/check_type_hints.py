from inspect import getfullargspec, isclass, isfunction
import transformers
from typing import get_type_hints


def is_call_overridden(cls):
    return "call" in cls.__dict__ and isfunction(cls.__dict__["call"])


def compute_missing_hints_tf(model):
    if isclass(model) and issubclass(model, transformers.TFPreTrainedModel) and is_call_overridden(model):
        actual_hints = set(get_type_hints(model.call))
        expected_hints = set(getfullargspec(model.call).args)
        expected_hints.remove("self")  # self does not carry type hints
        expected_hints.add("return")  # we need a type hint also for the output

        missing_hints = expected_hints - actual_hints
        if missing_hints:
            print(f"{obj}: {missing_hints}")


def compute_missing_hints_pytorch(model):
    if isclass(model) and issubclass(model, transformers.PreTrainedModel):
        actual_hints = set(get_type_hints(model.forward))
        expected_hints = set(getfullargspec(model.forward).args)
        expected_hints.remove("self")  # self does not carry type hints
        expected_hints.add("return")  # we need a type hint also for the output

        missing_hints = expected_hints - actual_hints
        if missing_hints:
            print(f"{obj}: {missing_hints}")


if __name__ == "__main__":
    # print("Checking type hints for PyTorch models")
    # for obj in dir(transformers):
    #     model = getattr(transformers, obj)
    #     compute_missing_hints_pytorch(model)

    print(50 * "-")
    print("Checking type hints for TensorFlow models")
    for obj in dir(transformers):
        model = getattr(transformers, obj)
        compute_missing_hints_tf(model)
