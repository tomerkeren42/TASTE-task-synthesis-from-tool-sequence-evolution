"""Airline domain validator — extracted from task_validator.py."""

import json
from typing import Any, Dict, List

from src.common.domain_validators.base import BaseDomainValidator
from src.common.flight_ambiguity_checker import find_gt_flight_conflicts


class AirlineValidator(BaseDomainValidator):
    """Airline-specific preflight and schema validation."""

    def __init__(self, domain_config):
        super().__init__(domain_config)
        self.valid_airports = set(domain_config.constants.get("valid_airports", []))
        self.dynamic_res_ids = domain_config.constants.get("dynamic_entity_ids", {}).get("reservations", [])
        self.system_limits = domain_config.constants.get("system_limits", {})

    def check_preflight(
        self, action_arguments: List[Dict[str, Any]], db_entities: Dict[str, Any]
    ) -> List[str]:
        """Check that all entity IDs in action arguments exist in the DB.

        Returns a list of error strings (empty if all checks pass).
        """
        errors: List[str] = []

        users = db_entities.get("users", {})
        reservations = db_entities.get("reservations", {})
        flights = db_entities.get("flights", {})

        user_ids = set(users.keys())
        reservation_ids = set(reservations.keys())

        # Track reservation IDs that will be dynamically created by book_reservation.
        # tau2-bench assigns IDs from a fixed pool: HATHAT, HATHAU, HATHAV (in order).
        dynamic_res_idx = 0  # next dynamic ID to assign
        # Map from LLM-chosen placeholder ID -> actual dynamic ID
        dynamic_id_map: Dict[str, str] = {}

        # First pass: identify book_reservation actions and map their expected
        # cancel IDs to the actual dynamic IDs that tau2-bench will assign.
        for action in action_arguments:
            if action.get("name") == "book_reservation":
                if dynamic_res_idx < len(self.dynamic_res_ids):
                    actual_id = self.dynamic_res_ids[dynamic_res_idx]
                    # If the action has an info.reservation_id hint, map it
                    info = action.get("info")
                    if isinstance(info, dict) and "reservation_id" in info:
                        placeholder = info["reservation_id"]
                        dynamic_id_map[placeholder] = actual_id
                    dynamic_res_idx += 1

        # All dynamic IDs that will be created (regardless of info hints)
        dynamic_reservation_ids = set(self.dynamic_res_ids[:dynamic_res_idx])
        # Also include any explicitly mapped IDs
        dynamic_reservation_ids.update(dynamic_id_map.values())

        # Load tool spec to build argument whitelist
        with open(self.domain_config.tool_spec_path) as _f:
            _tool_spec = json.load(_f)
        tool_allowed_args: Dict[str, set] = {
            tool_name: set(tool_data.get("parameters", {}).keys())
            for tool_name, tool_data in _tool_spec.items()
        }

        # --- system limit checks (entire action sequence) ---
        book_reservation_count = sum(
            1 for a in action_arguments if a.get("name") == "book_reservation"
        )
        if book_reservation_count > 3:
            errors.append(
                f"book_reservation called {book_reservation_count} times, "
                f"but the system limit is 3 (IDs: HATHAT, HATHAU, HATHAV)"
            )
        send_certificate_count = sum(
            1 for a in action_arguments if a.get("name") == "send_certificate"
        )
        if send_certificate_count > 3:
            errors.append(
                f"send_certificate called {send_certificate_count} times, "
                f"but the system limit is 3"
            )

        # Build set of (flight_number, date) pairs that exist and are available
        flight_dates: Dict[str, set] = {}  # flight_number -> set of dates
        available_flight_dates: Dict[str, set] = {}  # only "available" status
        for fn, flight_data in flights.items():
            dates = flight_data.get("dates", {})
            flight_dates[fn] = set(dates.keys())
            available_flight_dates[fn] = {
                d for d, info in dates.items()
                if info.get("status") == "available"
            }

        # Build reverse mapping: reservation_id -> owning user_id
        reservation_owners: Dict[str, str] = {}
        for rid, rdata in reservations.items():
            reservation_owners[rid] = rdata.get("user_id", "")

        # Build set of reservation IDs each user claims
        user_reservations: Dict[str, set] = {}
        for uid, udata in users.items():
            user_reservations[uid] = set(udata.get("reservations", []))

        for action in action_arguments:
            name = action.get("name", "")
            args = action.get("arguments", {})

            # --- argument whitelist check ---
            if name in tool_allowed_args:
                allowed = tool_allowed_args[name]
                extra_args = set(args.keys()) - allowed
                if extra_args:
                    errors.append(
                        f"Action '{name}': unexpected argument(s) {sorted(extra_args)}. "
                        f"Allowed arguments: {sorted(allowed)}"
                    )
                missing_args = allowed - set(args.keys())
                if missing_args and name != "list_all_airports":
                    errors.append(
                        f"Action '{name}': missing required argument(s) {sorted(missing_args)}"
                    )

            # --- airport code checks ---
            if name in ("book_reservation", "search_direct_flight",
                        "search_onestop_flight", "update_reservation_flights"):
                for field in ("origin", "destination"):
                    code = args.get(field)
                    if isinstance(code, str) and code not in self.valid_airports:
                        errors.append(
                            f"Action '{name}': {field} '{code}' is not a valid airport. "
                            f"Must be one of: {sorted(self.valid_airports)}"
                        )

            # --- user_id checks ---
            if name in ("get_user_details", "book_reservation", "send_certificate"):
                uid = args.get("user_id")
                if uid and uid not in user_ids:
                    errors.append(
                        f"Action '{name}': user_id '{uid}' not found in DB. "
                        f"Available: {sorted(user_ids)}"
                    )

            # --- reservation_id checks ---
            if name in (
                "get_reservation_details", "cancel_reservation",
                "update_reservation_flights", "update_reservation_baggages",
                "update_reservation_passengers",
            ):
                rid = args.get("reservation_id")
                if rid and rid not in reservation_ids:
                    # Check if this references a dynamically-created reservation
                    if rid in dynamic_id_map:
                        actual = dynamic_id_map[rid]
                        errors.append(
                            f"Action '{name}': reservation_id '{rid}' is a placeholder — "
                            f"book_reservation creates IDs from a FIXED pool: {self.dynamic_res_ids}. "
                            f"Use '{actual}' instead of '{rid}'."
                        )
                    elif rid in dynamic_reservation_ids:
                        pass  # correctly references a dynamic ID
                    else:
                        errors.append(
                            f"Action '{name}': reservation_id '{rid}' not found in DB. "
                            f"Available: {sorted(reservation_ids)}. "
                            f"Note: book_reservation creates IDs from a FIXED pool: {self.dynamic_res_ids}"
                        )
                elif rid:
                    # Check reservation is owned by a user that lists it
                    owner = reservation_owners.get(rid)
                    if owner and owner in user_reservations:
                        if rid not in user_reservations[owner]:
                            errors.append(
                                f"Action '{name}': reservation '{rid}' not in "
                                f"user '{owner}' reservations array"
                            )

            # --- flight checks (for booking and flight modification) ---
            if name in ("book_reservation", "update_reservation_flights"):
                flight_legs = args.get("flights", [])
                for leg in flight_legs:
                    fn = leg.get("flight_number", "")
                    date = leg.get("date", "")
                    if fn and fn not in flights:
                        errors.append(
                            f"Action '{name}': flight '{fn}' not found in DB. "
                            f"Available: {sorted(flights.keys())}"
                        )
                    elif fn and date and date not in flight_dates.get(fn, set()):
                        errors.append(
                            f"Action '{name}': flight '{fn}' has no entry for date '{date}'. "
                            f"Available dates: {sorted(flight_dates.get(fn, set()))}"
                        )
                    elif fn and date:
                        if date not in available_flight_dates.get(fn, set()):
                            actual_status = flights.get(fn, {}).get("dates", {}).get(date, {}).get("status", "?")
                            errors.append(
                                f"Action '{name}': flight '{fn}' on '{date}' has status "
                                f"'{actual_status}', expected 'available'"
                            )

            # --- payment_id checks ---
            if name in ("update_reservation_flights", "update_reservation_baggages"):
                pid = args.get("payment_id")
                if not pid:
                    errors.append(
                        f"Action '{name}': payment_id is missing or null"
                    )
                else:
                    # Find which user owns this reservation to check their payment methods
                    rid = args.get("reservation_id", "")
                    owner = reservation_owners.get(rid, "")
                    if owner and owner in users:
                        payment_methods = users[owner].get("payment_methods", {})
                        if pid not in payment_methods:
                            errors.append(
                                f"Action '{name}': payment_id '{pid}' not found in "
                                f"user '{owner}' payment_methods. "
                                f"Available: {sorted(payment_methods.keys())}"
                            )

            if name == "book_reservation":
                payment_methods_arg = args.get("payment_methods", [])
                uid = args.get("user_id", "")
                if uid and uid in users:
                    user_payments = users[uid].get("payment_methods", {})
                    for pm in payment_methods_arg:
                        pid = pm.get("payment_id", "")
                        if pid and pid not in user_payments:
                            errors.append(
                                f"Action '{name}': payment_id '{pid}' not found in "
                                f"user '{uid}' payment_methods. "
                                f"Available: {sorted(user_payments.keys())}"
                            )

        # --- certificate reuse check (across all book_reservation actions) ---
        # Certificates are removed after a single book_reservation call.
        certificate_uses: Dict[str, int] = {}  # payment_id -> number of book_reservation uses
        for action in action_arguments:
            if action.get("name") == "book_reservation":
                for pm in action.get("arguments", {}).get("payment_methods", []):
                    pid = pm.get("payment_id", "")
                    # Check if this payment is a certificate in any user's payment_methods
                    for uid, udata in users.items():
                        user_pms = udata.get("payment_methods", {})
                        if pid in user_pms:
                            source = user_pms[pid].get("source", "")
                            if source == "certificate":
                                certificate_uses[pid] = certificate_uses.get(pid, 0) + 1
        for pid, count in certificate_uses.items():
            if count > 1:
                errors.append(
                    f"Certificate '{pid}' is used in {count} book_reservation actions, "
                    f"but certificates are REMOVED after a single use. "
                    f"Each booking must use a different certificate or a credit card."
                )

            # --- passenger count check ---
            if name == "update_reservation_passengers":
                rid = args.get("reservation_id", "")
                new_passengers = args.get("passengers", [])
                if rid and rid in reservations:
                    existing_passengers = reservations[rid].get("passengers", [])
                    if len(new_passengers) != len(existing_passengers):
                        errors.append(
                            f"Action '{name}': passenger count mismatch for '{rid}': "
                            f"sending {len(new_passengers)} but reservation has {len(existing_passengers)}"
                        )

        # --- flight ambiguity check (multiple flights on same route+date) ---
        flight_conflicts = find_gt_flight_conflicts(action_arguments, flights)
        for conflict in flight_conflicts:
            competitors = ", ".join(conflict["competing_flights"])
            if conflict["gt_flight"] == "search":
                # For search actions there is no specific GT flight — the fix is to
                # leave exactly ONE flight on the route+date.  The old message listed
                # ALL flights as "competing", causing the patch LLM to remove every
                # flight on the route (breaking the search), then re-adding them.
                keep = conflict.get("keep_flight", "")
                errors.append(
                    f"Flight ambiguity: search action on "
                    f"{conflict['origin']}->{conflict['destination']} date {conflict['date']} "
                    f"finds multiple flights. "
                    f"Keep exactly ONE flight (keep '{keep}') and DELETE the extra flight(s) [{competitors}] "
                    f"from the DB entirely. Do NOT remove all flights — the search must find exactly one."
                )
            else:
                errors.append(
                    f"Flight ambiguity: GT expects flight '{conflict['gt_flight']}' on "
                    f"{conflict['origin']}->{conflict['destination']} date {conflict['date']}, "
                    f"but competing flight(s) [{competitors}] serve the same route+date. "
                    f"Remove competing flights to avoid evaluation ambiguity."
                )

        return errors

    def validate_db_schema(self, db_entities: Dict[str, Any]) -> List[str]:
        """Validate DB entities against the FlightDB Pydantic schema.

        Catches common LLM mistakes and returns clear, actionable error messages
        instead of the raw Pydantic discriminated-union explosion (which can
        produce 60-120 confusing errors for a single wrong field).
        """
        errors: List[str] = []

        # -- Users --
        for uid, user_data in db_entities.get("users", {}).items():
            if not isinstance(user_data, dict):
                errors.append(f"User '{uid}': expected a dict, got {type(user_data).__name__}")
                continue

            # name must be a nested object {first_name, last_name}
            name = user_data.get("name")
            if name is None:
                if "first_name" in user_data or "last_name" in user_data:
                    errors.append(
                        f"User '{uid}': 'name' must be a nested object "
                        f'{{"first_name": "...", "last_name": "..."}}, '
                        f"not top-level first_name/last_name fields"
                    )
                else:
                    errors.append(f"User '{uid}': missing required field 'name'")
            elif not isinstance(name, dict):
                errors.append(f"User '{uid}': 'name' must be an object, got {type(name).__name__}")

            # address -- all fields required, no nulls
            address = user_data.get("address")
            if address is None:
                errors.append(f"User '{uid}': missing required field 'address'")
            elif isinstance(address, dict):
                # Check for wrong field names
                if "street" in address and "address1" not in address:
                    errors.append(
                        f"User '{uid}': address uses 'street' but the schema requires 'address1'"
                    )
                if "zip_code" in address and "zip" not in address:
                    errors.append(
                        f"User '{uid}': address uses 'zip_code' but the schema requires 'zip'"
                    )
                for field in ("address1", "city", "state", "country", "zip"):
                    val = address.get(field)
                    if val is None:
                        errors.append(
                            f"User '{uid}': address.{field} must be a non-null string. "
                            f"For non-US addresses use a region code or district name, never null."
                        )

            # payment_methods -- validate each payment type's required fields
            for pid, pm in user_data.get("payment_methods", {}).items():
                if not isinstance(pm, dict):
                    continue
                source = pm.get("source")
                if source is None:
                    errors.append(
                        f"User '{uid}' payment '{pid}': missing required field 'source' "
                        f"(must be 'credit_card', 'gift_card', or 'certificate')"
                    )
                    continue
                _VALID_PAYMENT_SOURCES = {"credit_card", "gift_card", "certificate"}
                if source not in _VALID_PAYMENT_SOURCES:
                    errors.append(
                        f"User '{uid}' payment '{pid}': source '{source}' is invalid. "
                        f"Must be one of: {sorted(_VALID_PAYMENT_SOURCES)}"
                    )
                    continue
                if "id" not in pm:
                    errors.append(
                        f"User '{uid}' payment '{pid}': missing required field 'id'"
                    )
                if source == "credit_card":
                    for field in ("brand", "last_four"):
                        if field not in pm:
                            # Check for common LLM-hallucinated alternatives
                            alternatives = {
                                "brand": ["card_type"],
                                "last_four": ["last_digits", "last4"],
                            }
                            found_alt = None
                            for alt in alternatives.get(field, []):
                                if alt in pm:
                                    found_alt = alt
                                    break
                            hint = f" (found '{found_alt}' — rename it to '{field}')" if found_alt else ""
                            errors.append(
                                f"User '{uid}' payment '{pid}': credit_card missing "
                                f"required field '{field}'{hint}"
                            )
                elif source == "gift_card":
                    if "amount" not in pm:
                        errors.append(
                            f"User '{uid}' payment '{pid}': gift_card missing required field 'amount'"
                        )
                elif source == "certificate":
                    if "amount" not in pm:
                        errors.append(
                            f"User '{uid}' payment '{pid}': certificate missing required field 'amount'"
                        )

        # -- Flights --
        for fn, flight_data in db_entities.get("flights", {}).items():
            if not isinstance(flight_data, dict):
                continue
            for field in ("scheduled_departure_time_est", "scheduled_arrival_time_est"):
                if field not in flight_data:
                    errors.append(f"Flight '{fn}': missing required field '{field}'")
            # Airport code validation
            for field in ("origin", "destination"):
                code = flight_data.get(field)
                if isinstance(code, str) and code not in self.valid_airports:
                    errors.append(
                        f"Flight '{fn}': {field} '{code}' is not a valid airport. "
                        f"Must be one of: {sorted(self.valid_airports)}"
                    )
            for date_str, date_info in flight_data.get("dates", {}).items():
                if not isinstance(date_info, dict):
                    continue
                status = date_info.get("status")
                if status == "available":
                    seats = date_info.get("available_seats")
                    if not isinstance(seats, dict):
                        errors.append(
                            f"Flight '{fn}' date '{date_str}': 'available_seats' must be "
                            f'a dict like {{"economy": 50, "business": 10}}, '
                            f"got {type(seats).__name__}: {seats}"
                        )
                    prices = date_info.get("prices")
                    if not isinstance(prices, dict):
                        errors.append(
                            f"Flight '{fn}' date '{date_str}': 'prices' must be "
                            f'a dict like {{"economy": 350, "business": 800}}, '
                            f"got {type(prices).__name__}"
                        )
                    if isinstance(prices, dict):
                        for cabin_class, price_val in prices.items():
                            if isinstance(price_val, float) and not price_val.is_integer():
                                errors.append(
                                    f"Flight '{fn}' date '{date_str}': price for '{cabin_class}' "
                                    f"is {price_val} (float) — must be an integer"
                                )
                            elif isinstance(price_val, float):
                                errors.append(
                                    f"Flight '{fn}' date '{date_str}': price for '{cabin_class}' "
                                    f"is {price_val} (float) — use {int(price_val)} instead"
                                )
                elif status in ("landed",):
                    for tf in ("actual_departure_time_est", "actual_arrival_time_est"):
                        if tf not in date_info:
                            errors.append(f"Flight '{fn}' date '{date_str}': status 'landed' requires '{tf}'")
                elif status in ("delayed",):
                    for tf in ("estimated_departure_time_est", "estimated_arrival_time_est"):
                        if tf not in date_info:
                            errors.append(f"Flight '{fn}' date '{date_str}': status 'delayed' requires '{tf}'")
                elif status in ("flying",):
                    for tf in ("actual_departure_time_est", "estimated_arrival_time_est"):
                        if tf not in date_info:
                            errors.append(f"Flight '{fn}' date '{date_str}': status 'flying' requires '{tf}'")
                elif status in ("on time",):
                    for tf in ("estimated_departure_time_est", "estimated_arrival_time_est"):
                        if tf not in date_info:
                            errors.append(f"Flight '{fn}' date '{date_str}': status 'on time' requires '{tf}'")
                elif status == "cancelled":
                    pass  # no extra fields needed
                elif status is not None:
                    errors.append(
                        f"Flight '{fn}' date '{date_str}': unknown status '{status}'. "
                        f"Must be one of: available, landed, cancelled, delayed, flying, on time"
                    )

        # -- Reservations --
        for rid, res_data in db_entities.get("reservations", {}).items():
            if not isinstance(res_data, dict):
                continue
            # Airport code validation on reservation origin/destination
            for field in ("origin", "destination"):
                code = res_data.get(field)
                if isinstance(code, str) and code not in self.valid_airports:
                    errors.append(
                        f"Reservation '{rid}': {field} '{code}' is not a valid airport. "
                        f"Must be one of: {sorted(self.valid_airports)}"
                    )
            flights_list = res_data.get("flights", [])
            for i, flight_leg in enumerate(flights_list):
                if isinstance(flight_leg, dict):
                    for field in ("origin", "destination", "price"):
                        if field not in flight_leg:
                            errors.append(
                                f"Reservation '{rid}' flight[{i}]: missing required field '{field}'. "
                                f"Each flight leg needs: flight_number, date, origin, destination, price"
                            )
                    # Airport code validation on flight legs
                    for field in ("origin", "destination"):
                        code = flight_leg.get(field)
                        if isinstance(code, str) and code not in self.valid_airports:
                            errors.append(
                                f"Reservation '{rid}' flight[{i}]: {field} '{code}' is not a valid airport. "
                                f"Must be one of: {sorted(self.valid_airports)}"
                            )
                    price_val = flight_leg.get("price")
                    if isinstance(price_val, float):
                        if price_val.is_integer():
                            errors.append(
                                f"Reservation '{rid}' flight[{i}]: price is {price_val} "
                                f"(float) — use {int(price_val)} instead"
                            )
                        else:
                            errors.append(
                                f"Reservation '{rid}' flight[{i}]: price is {price_val} "
                                f"(float) — must be an integer"
                            )

        return errors
