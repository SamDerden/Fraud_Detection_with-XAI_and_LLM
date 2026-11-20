"""Webinar 15 practice: unique value pairs with O(n) expected time, no input mutation."""

def find_pairs(numbers: list[int], target: int) -> list[tuple[int,int]]:
    """Return unique ascending-value pairs (a,b) with a+b == target.

    Pairs are deduplicated by values; (3,3) requires two occurrences.
    Order is deterministic and sorted lexicographically.
    """
    seen=set();pairs=set()
    for value in numbers:
        other=target-value
        if other in seen:
            pairs.add((min(value,other),max(value,other)))
        seen.add(value)
    return sorted(pairs)
