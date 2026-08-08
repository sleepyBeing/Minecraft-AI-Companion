"""Cover geometry and dynamic safe-position planning for Stage Four."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


CoverRectangle = tuple[float, float, float, float]

COVER_RECTANGLES: tuple[CoverRectangle, ...] = (
    (7.0, 8.0, 1.0, 6.0),
    (7.0, 8.0, 9.0, 14.0),
)
SAFE_SIDE_CLEARANCE = 1.25
END_CLEARANCE = 0.75
END_MARGIN = 1.25
DYNAMIC_SHIFT_GAIN = 0.35
DYNAMIC_SHIFT_LIMIT = 1.0


@dataclass(frozen=True)
class CoverPlan:
    cover_index: int
    navigation_position: np.ndarray
    protected_position: np.ndarray


def choose_cover_plan(
    bot: np.ndarray,
    target: np.ndarray,
    *,
    room_size: float,
) -> CoverPlan:
    plans = [
        plan_for_rectangle(
            bot,
            target,
            rectangle,
            cover_index=index,
            room_size=room_size,
        )
        for index, rectangle in enumerate(COVER_RECTANGLES)
    ]
    return min(
        plans,
        key=lambda plan: (
            float(np.linalg.norm(plan.navigation_position - bot))
            + 0.25
            * float(
                np.linalg.norm(
                    plan.protected_position - plan.navigation_position
                )
            )
        ),
    )


def plan_for_rectangle(
    bot: np.ndarray,
    target: np.ndarray,
    rectangle: CoverRectangle,
    *,
    cover_index: int,
    room_size: float,
) -> CoverPlan:
    protected = protected_position(rectangle, target, room_size=room_size)
    navigation = navigation_position(
        bot,
        target,
        protected,
        rectangle,
        room_size=room_size,
    )
    return CoverPlan(
        cover_index=cover_index,
        navigation_position=navigation,
        protected_position=protected,
    )


def protected_position(
    rectangle: CoverRectangle,
    target: np.ndarray,
    *,
    room_size: float,
) -> np.ndarray:
    min_x, max_x, min_z, max_z = rectangle
    center_x = (min_x + max_x) / 2.0
    center_z = (min_z + max_z) / 2.0
    width = max_x - min_x
    depth = max_z - min_z

    if depth >= width:
        safe_x = (
            min_x - SAFE_SIDE_CLEARANCE
            if target[0] >= center_x
            else max_x + SAFE_SIDE_CLEARANCE
        )
        available_shift = max(0.0, depth / 2.0 - END_MARGIN)
        shift = float(
            np.clip(
                (center_z - target[1]) * DYNAMIC_SHIFT_GAIN,
                -min(DYNAMIC_SHIFT_LIMIT, available_shift),
                min(DYNAMIC_SHIFT_LIMIT, available_shift),
            )
        )
        point = np.array([safe_x, center_z + shift], dtype=np.float32)
    else:
        safe_z = (
            min_z - SAFE_SIDE_CLEARANCE
            if target[1] >= center_z
            else max_z + SAFE_SIDE_CLEARANCE
        )
        available_shift = max(0.0, width / 2.0 - END_MARGIN)
        shift = float(
            np.clip(
                (center_x - target[0]) * DYNAMIC_SHIFT_GAIN,
                -min(DYNAMIC_SHIFT_LIMIT, available_shift),
                min(DYNAMIC_SHIFT_LIMIT, available_shift),
            )
        )
        point = np.array([center_x + shift, safe_z], dtype=np.float32)

    return np.clip(point, 0.5, room_size - 0.5).astype(np.float32)


def navigation_position(
    bot: np.ndarray,
    target: np.ndarray,
    protected: np.ndarray,
    rectangle: CoverRectangle,
    *,
    room_size: float,
) -> np.ndarray:
    min_x, max_x, min_z, max_z = rectangle
    center_x = (min_x + max_x) / 2.0
    center_z = (min_z + max_z) / 2.0
    vertical_wall = (max_z - min_z) >= (max_x - min_x)

    if vertical_wall:
        bot_side = _side(float(bot[0] - center_x))
        protected_side = _side(float(protected[0] - center_x))
        if bot_side == protected_side:
            return protected.copy()

        route_coordinate = (
            max_z + END_CLEARANCE
            if target[1] < center_z
            else min_z - END_CLEARANCE
        )
        route_coordinate = float(np.clip(route_coordinate, 0.5, room_size - 0.5))
        bot_lane = (
            min_x - SAFE_SIDE_CLEARANCE
            if bot_side < 0
            else max_x + SAFE_SIDE_CLEARANCE
        )
        if min_z <= bot[1] <= max_z:
            point = np.array([bot_lane, route_coordinate], dtype=np.float32)
        else:
            point = np.array([protected[0], route_coordinate], dtype=np.float32)
    else:
        bot_side = _side(float(bot[1] - center_z))
        protected_side = _side(float(protected[1] - center_z))
        if bot_side == protected_side:
            return protected.copy()

        route_coordinate = (
            max_x + END_CLEARANCE
            if target[0] < center_x
            else min_x - END_CLEARANCE
        )
        route_coordinate = float(np.clip(route_coordinate, 0.5, room_size - 0.5))
        bot_lane = (
            min_z - SAFE_SIDE_CLEARANCE
            if bot_side < 0
            else max_z + SAFE_SIDE_CLEARANCE
        )
        if min_x <= bot[0] <= max_x:
            point = np.array([route_coordinate, bot_lane], dtype=np.float32)
        else:
            point = np.array([route_coordinate, protected[1]], dtype=np.float32)

    return np.clip(point, 0.5, room_size - 0.5).astype(np.float32)


def is_geometrically_occluded(bot: np.ndarray, target: np.ndarray) -> bool:
    return any(
        segment_intersects_rectangle(bot, target, rectangle)
        for rectangle in COVER_RECTANGLES
    )


def point_inside_cover(point: np.ndarray) -> bool:
    return any(
        min_x <= point[0] <= max_x and min_z <= point[1] <= max_z
        for min_x, max_x, min_z, max_z in COVER_RECTANGLES
    )


def segment_intersects_rectangle(
    start: np.ndarray,
    end: np.ndarray,
    rectangle: CoverRectangle,
) -> bool:
    min_x, max_x, min_z, max_z = rectangle
    delta = end - start
    minimum = 0.0
    maximum = 1.0
    for origin, change, low, high in (
        (start[0], delta[0], min_x, max_x),
        (start[1], delta[1], min_z, max_z),
    ):
        if abs(float(change)) < 1e-9:
            if origin < low or origin > high:
                return False
            continue
        first = (low - origin) / change
        second = (high - origin) / change
        minimum = max(minimum, float(min(first, second)))
        maximum = min(maximum, float(max(first, second)))
        if minimum > maximum:
            return False
    return True


def _side(value: float) -> int:
    return -1 if value < 0.0 else 1
