"""Small shared click helpers for project CLIs."""

from collections.abc import Callable
from typing import TypeVar

CLICK_CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"],
}

F = TypeVar("F", bound=Callable[..., object])


def apply_click_options(*options: Callable[[F], F]) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        for option in reversed(options):
            func = option(func)
        return func

    return decorator
