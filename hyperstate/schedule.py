import math
from typing import (
    Callable,
    List,
    Any,
    TypeVar,
)
from dataclasses import dataclass

C = TypeVar("C")
S = TypeVar("S")
T = TypeVar("T")


INTERPOLATORS = {
    "lin": lambda x: 1 - x,
    "cos": lambda x: math.cos(x * math.pi) * 0.5 + 0.5,
    "step": lambda _: 1,
}


@dataclass
class Schedule:
    update_value: Callable[[Any], None]
    unparsed: str


@dataclass
class ScheduleSegment:
    start: float
    end: float
    initial_value: float
    final_value: float
    interpolator: Callable[[float], float]


@dataclass
class PiecewiseFunction:
    segments: List[ScheduleSegment]
    xname: str

    def get_value(self, x: float) -> float:
        segment = None
        for s in self.segments:
            segment = s
            if x < s.end:
                break
        rescaled = (x - segment.start) / (segment.end - segment.start)
        if rescaled < 0:
            rescaled = 0
        elif rescaled > 1:
            rescaled = 1

        w = segment.interpolator(rescaled)
        return w * segment.initial_value + (1 - w) * segment.final_value


def _parse_schedule(schedule: str) -> Callable[[float], float]:
    # Grammar
    #
    # rule := ident ": " point [join point]
    # join := " " | " " ident " "
    # point := num "@" num
    # ident := "lin" | "cos" | "exp" | "step" | "quad" | "poly(" num ")"
    # num := `float` | `int` | `path`
    # path := ident [ "." path]
    #
    # Example:
    # "0.3@step=0 lin 0.15@150e6 0.1@200e6 0.01@250e6"

    # TODO: lots of checks and helpful error messages
    parts = schedule.split(" ")
    interpolator = INTERPOLATORS["lin"]
    last_x, last_y = None, None
    xname = parts[0][:-1]
    segments = []
    for part in parts[1:]:
        if "@" in part:
            y, x = part.split("@")
            y, x = float(y), float(x)
            if last_x is not None:
                segments.append(
                    ScheduleSegment(
                        start=last_x,
                        initial_value=last_y,
                        end=x,
                        final_value=y,
                        interpolator=interpolator,
                    )
                )
            last_x, last_y = x, y
        else:
            interpolator = INTERPOLATORS[part]
    return PiecewiseFunction(segments, xname)
