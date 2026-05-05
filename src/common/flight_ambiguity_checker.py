"""Detect flights in a task DB that conflict with GT action expectations.

A conflict occurs when the DB has multiple flight entities serving the same
(origin, destination, date) and one of those flights is referenced by a GT action.
This causes ambiguity: the simulated user may request a different flight than the
GT expects, leading to false evaluation failures.
"""
from typing import Any, Dict, List

_FLIGHT_REF_ACTIONS = {"book_reservation", "update_reservation_flights"}
_SEARCH_ACTIONS = {"search_direct_flight", "search_onestop_flight"}


def find_gt_flight_conflicts(
    gt_actions: List[Dict[str, Any]],
    db_flights: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Find DB flights that conflict with GT-referenced flights.

    Checks two types of GT actions:
    - Direct references (book_reservation, update_reservation_flights): extract
      flight_number, look up route in DB.
    - Search actions (search_direct_flight, search_onestop_flight): extract
      origin, destination, date directly from args.

    Returns:
        List of conflict dicts with gt_flight, date, origin, destination, competing_flights.
    """
    route_date_index: Dict[tuple, set] = {}
    for fn, fdata in db_flights.items():
        origin = fdata.get("origin", "")
        dest = fdata.get("destination", "")
        for date_str in fdata.get("dates", {}):
            key = (origin, dest, date_str)
            route_date_index.setdefault(key, set()).add(fn)

    conflicts = []
    seen = set()

    # Pass 1: Direct flight references
    for action in gt_actions:
        if action.get("name") not in _FLIGHT_REF_ACTIONS:
            continue
        for leg in action.get("arguments", {}).get("flights", []):
            fn = leg.get("flight_number", "")
            date = leg.get("date", "")
            if not fn or not date:
                continue
            fdata = db_flights.get(fn)
            if not fdata:
                continue
            origin = fdata.get("origin", "")
            dest = fdata.get("destination", "")
            key = (origin, dest, date)
            if key in seen:
                continue
            seen.add(key)
            competitors = route_date_index.get(key, set()) - {fn}
            if competitors:
                conflicts.append({
                    "gt_flight": fn,
                    "date": date,
                    "origin": origin,
                    "destination": dest,
                    "competing_flights": sorted(competitors),
                })

    # Pass 2: Search actions
    for action in gt_actions:
        if action.get("name") not in _SEARCH_ACTIONS:
            continue
        args = action.get("arguments", {})
        origin = args.get("origin", "")
        dest = args.get("destination", "")
        date = args.get("date", "")
        if not origin or not dest or not date:
            continue
        key = (origin, dest, date)
        if key in seen:
            continue
        seen.add(key)
        flights_on_route = route_date_index.get(key, set())
        if len(flights_on_route) > 1:
            all_sorted = sorted(flights_on_route)
            # Keep the first flight; the rest are extra and should be removed.
            # Listing ALL flights as "competing" caused the patch LLM to remove
            # every flight (leaving none), breaking the search action.
            extras = all_sorted[1:]
            conflicts.append({
                "gt_flight": "search",
                "date": date,
                "origin": origin,
                "destination": dest,
                "keep_flight": all_sorted[0],
                "competing_flights": extras,
            })

    return conflicts
