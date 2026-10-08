from __future__ import annotations

import math

import numpy as np
import pytest

from gameplay_events.hud import CoinCounter, CoinRead


FRAME = np.zeros((480, 854, 3), dtype=np.uint8)


def feed(counter: CoinCounter, values: list[int | None], start: float = 0.0):
    observations = iter(values)

    def fake_read(_frame):
        value = next(observations)
        if value is None:
            return None
        text = str(value)
        return CoinRead(value, 0.9, (820, 7, 847, 19), text)

    counter.read = fake_read  # type: ignore[method-assign]
    events = []
    for index in range(len(values)):
        events.extend(counter.update(FRAME, start + index * 0.1))
    return events


def test_one_frame_misread_does_not_change_baseline():
    counter = CoinCounter(confirm_frames=2, confirm_seconds=0.1)
    assert feed(counter, [10, 10]) == []
    assert counter.last_count == 10

    assert feed(counter, [11, 10], start=1.0) == []
    assert counter.last_count == 10


def test_confirmed_increase_and_decrease_emit_first_supporting_time():
    counter = CoinCounter(confirm_frames=2, confirm_seconds=0.1)
    feed(counter, [10, 10])

    increased = feed(counter, [12, 12], start=1.0)
    assert len(increased) == 1
    assert increased[0].event == "coin_count_changed"
    assert increased[0].timestamp == pytest.approx(1.0)
    assert increased[0].confirmed_at == pytest.approx(1.1)
    assert increased[0].evidence["old_count"] == 10
    assert increased[0].evidence["new_count"] == 12
    assert increased[0].evidence["delta"] == 2

    decreased = feed(counter, [9, 9], start=2.0)
    assert len(decreased) == 1
    assert decreased[0].evidence["delta"] == -3
    assert counter.last_count == 9


def test_reset_discards_baseline_and_pending_state():
    counter = CoinCounter(confirm_frames=2, confirm_seconds=0.1)
    feed(counter, [10, 10])
    feed(counter, [11], start=1.0)
    counter.reset()
    assert counter.last_count is None

    # A stable post-reset value establishes a fresh baseline, not an event.
    assert feed(counter, [30, 30], start=0.0) == []
    assert counter.last_count == 30


@pytest.mark.parametrize("bad_timestamp", [-1.0, math.nan, math.inf, -math.inf])
def test_invalid_timestamp_is_rejected(bad_timestamp):
    counter = CoinCounter()
    with pytest.raises(ValueError):
        counter.update(FRAME, bad_timestamp)


def test_timestamp_cannot_move_backwards():
    counter = CoinCounter(confirm_frames=1, confirm_seconds=0.0)
    counter.read = lambda _frame: None  # type: ignore[method-assign]
    counter.update(FRAME, 1.0)
    with pytest.raises(ValueError):
        counter.update(FRAME, 0.9)


def test_blank_frame_is_not_a_coin_read():
    counter = CoinCounter()
    assert counter.read(FRAME) is None


def test_frame_contract_is_checked():
    counter = CoinCounter()
    with pytest.raises(ValueError):
        counter.read(np.zeros((480, 854), dtype=np.uint8))
    with pytest.raises(ValueError):
        counter.read(np.zeros((480, 854, 3), dtype=np.float32))
