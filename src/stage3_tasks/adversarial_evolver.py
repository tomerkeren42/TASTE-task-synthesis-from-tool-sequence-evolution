"""
AdversarialEvolver: multi-phase adversarial task evolution.

Phase 1: Analyze golden actions + DB → structured adversarial strategy
Phase 2: Build decoy DB entities (flights, reservations) from strategy
Phase 3: Write adversarial user scenario using strategy + traps
"""
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from src.common.call_to_llm import LLMCaller
from src.common.llm_response_parser import LLMResponseParser
from src.common.prompt_manager import PromptManager
from src.common.tool_spec_retriever import ToolsSpecRetriever
from src.common.domain_utils import load_policy


class AdversarialEvolver:
    """
    Multi-phase adversarial task evolution.

    Transforms cooperative easy tasks into hard tasks by injecting
    adversarial user behavior, DB traps, and conditional escalation
    targeting wrong WRITE actions.
    """

    def __init__(
        self,
        model_name: str = "vertex_ai/gemini-3-flash-preview",
        domain: str = "airline",
        max_output_tokens: int = 32768,
        domain_config=None,
    ):
        self.caller = LLMCaller(
            model_name=model_name,
            max_output_tokens=max_output_tokens,
        )
        if domain_config is None:
            from src.common.domain_config import DomainConfig
            domain_config = DomainConfig(domain)
        self.prompt_manager = PromptManager(prompts_dir=domain_config.prompts_dir, domain=domain_config.domain)
        self.tool_spec = ToolsSpecRetriever(path=domain_config.tool_spec_path)
        self.parser = LLMResponseParser()
        self.action_types = self.tool_spec.get_action_types()
        self.policy = load_policy(domain)
        self.domain = domain
        self.db_schema_str = domain_config.get_db_schema_str()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _format_actions_with_types(self, actions: List[Dict[str, Any]]) -> str:
        """Format actions as 'action_name [READ/WRITE](args_json)'."""
        if not actions:
            return "(no actions)"
        parts = []
        for i, a in enumerate(actions):
            name = a.get("name", "unknown")
            action_type = self.action_types.get(name, "GENERIC")
            args = a.get("arguments", {})
            if args:
                args_str = json.dumps(args, indent=2)
                parts.append(f"[{i}] {name} [{action_type}]({args_str})")
            else:
                parts.append(f"[{i}] {name} [{action_type}]()")
        return "\n\n".join(parts)

    def _get_db_state(self, task_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Extract the DB entities from a task's initial_state."""
        return (
            task_dict.get("initial_state", {})
            .get("initialization_data", {})
            .get("agent_data", {})
        )

    def _extract_db_summary(self, task_dict: Dict[str, Any]) -> str:
        """Build a human-readable DB summary for Phase 3."""
        db = self._get_db_state(task_dict)
        lines = []

        if self.domain == "retail":
            # Users
            for uid, user in db.get("users", {}).items():
                name = user.get("name", {})
                lines.append(f"USER: {name.get('first_name', '')} {name.get('last_name', '')} ({uid})")
                lines.append(f"  Email: {user.get('email', '')}")
                lines.append(f"  Orders: {user.get('orders', [])}")
                for pid, pm in user.get("payment_methods", {}).items():
                    source = pm.get("source", "")
                    if source == "credit_card":
                        lines.append(f"  Payment: {pid} — credit card ({pm.get('brand','')}, last four {pm.get('last_four','')})")
                    elif source == "gift_card":
                        lines.append(f"  Payment: {pid} — gift card (balance: ${pm.get('balance', 0)})")
                    elif source == "paypal":
                        lines.append(f"  Payment: {pid} — paypal")

            # Orders
            for oid, order in db.get("orders", {}).items():
                lines.append(f"\nORDER: {oid} — status={order.get('status','')}")
                for item in order.get("items", []):
                    lines.append(f"  Item: {item.get('name','')} (product={item.get('product_id','')}, item_id={item.get('item_id','')}, ${item.get('price','')})")
                for ph in order.get("payment_history", []):
                    lines.append(f"  Payment: {ph.get('transaction_type','')} ${ph.get('amount','')} via {ph.get('payment_method_id','')}")
        elif self.domain == "telecom":
            # Telecom agent_data stores entities as LISTS of dicts, not dict-of-dicts.
            for cust in db.get("customers", []):
                lines.append(f"CUSTOMER: {cust.get('full_name', '')} ({cust.get('customer_id', '')})")
                lines.append(f"  Email: {cust.get('email', '')}")
                lines.append(f"  Phone: {cust.get('phone_number', '')}")
                lines.append(f"  Lines: {cust.get('line_ids', [])}")
                lines.append(f"  Bills: {cust.get('bill_ids', [])}")

            for line in db.get("lines", []):
                lines.append(f"\nLINE: {line.get('line_id', '')} — status={line.get('status', '')}")
                lines.append(f"  Phone: {line.get('phone_number', '')}")
                lines.append(f"  Plan: {line.get('plan_id', '')}")
                lines.append(f"  Device: {line.get('device_id', '')}")
                lines.append(f"  Roaming: {line.get('roaming_enabled', False)}")
                lines.append(f"  Data used: {line.get('data_used_gb', 0)} GB")
                lines.append(f"  Contract end: {line.get('contract_end_date', '')}")
                if line.get("suspension_start_date"):
                    lines.append(f"  Suspension start: {line.get('suspension_start_date')}")

            for bill in db.get("bills", []):
                lines.append(f"\nBILL: {bill.get('bill_id', '')} — status={bill.get('status', '')}")
                lines.append(f"  Customer: {bill.get('customer_id', '')}")
                lines.append(f"  Total due: ${bill.get('total_due', 0)}")
                lines.append(f"  Period: {bill.get('period_start', '')} to {bill.get('period_end', '')}")
                lines.append(f"  Due date: {bill.get('due_date', '')}")

            for device in db.get("devices", []):
                lines.append(f"\nDEVICE: {device.get('device_id', '')} — {device.get('model', '')}")
                lines.append(f"  Type: {device.get('device_type', '')}")
                lines.append(f"  eSIM: {device.get('is_esim_capable', False)}")
                lines.append(f"  Activated: {device.get('activated', False)}")

            for plan in db.get("plans", []):
                lines.append(f"\nPLAN: {plan.get('plan_id', '')} — {plan.get('name', '')}")
                lines.append(f"  Data: {plan.get('data_limit_gb', 0)} GB, ${plan.get('price_per_month', 0)}/mo")
                if plan.get("data_refueling_price_per_gb") is not None:
                    lines.append(f"  Refuel price: ${plan.get('data_refueling_price_per_gb')}/GB")

            # Surroundings (if present)
            surroundings = db.get("surroundings")
            if isinstance(surroundings, dict):
                lines.append(f"\nSURROUNDINGS:")
                lines.append(f"  Abroad: {surroundings.get('is_abroad', False)}")
                lines.append(f"  Signal: {surroundings.get('signal_strength', {})}")
                lines.append(f"  Line active: {surroundings.get('line_active', True)}")
        else:
            # Airline: Users
            for uid, user in db.get("users", {}).items():
                name = user.get("name", {})
                lines.append(f"USER: {name.get('first_name', '')} {name.get('last_name', '')} ({uid})")
                lines.append(f"  Membership: {user.get('membership', 'unknown')}")
                lines.append(f"  Reservations: {user.get('reservations', [])}")
                for pid, pm in user.get("payment_methods", {}).items():
                    source = pm.get("source", "")
                    if source == "credit_card":
                        lines.append(f"  Payment: {pid} — credit card ({pm.get('brand','')}, last four {pm.get('last_four','')})")
                    elif source == "gift_card":
                        lines.append(f"  Payment: {pid} — gift card (balance: ${pm.get('amount', 0)})")
                    elif source == "certificate":
                        lines.append(f"  Payment: {pid} — certificate (amount: ${pm.get('amount', 0)})")
                for sp in user.get("saved_passengers", []):
                    lines.append(f"  Saved passenger: {sp.get('first_name','')} {sp.get('last_name','')} (DOB: {sp.get('dob','')})")

            # Reservations
            for rid, res in db.get("reservations", {}).items():
                lines.append(f"\nRESERVATION: {rid}")
                lines.append(f"  Route: {res.get('origin','')} → {res.get('destination','')}, {res.get('flight_type','')}")
                lines.append(f"  Cabin: {res.get('cabin','')}")
                lines.append(f"  Insurance: {res.get('insurance','no')}")
                lines.append(f"  Created: {res.get('created_at','')}")
                lines.append(f"  Passengers: {len(res.get('passengers', []))}")
                for p in res.get("passengers", []):
                    lines.append(f"    - {p.get('first_name','')} {p.get('last_name','')} (DOB: {p.get('dob','')})")
                lines.append(f"  Flights:")
                for fl in res.get("flights", []):
                    lines.append(f"    - {fl.get('flight_number','')} {fl.get('origin','')}->{fl.get('destination','')} on {fl.get('date','')} (${fl.get('price','')})")
                lines.append(f"  Baggages: {res.get('total_baggages',0)} total, {res.get('nonfree_baggages',0)} non-free")

            # Flights
            for fnum, flight in db.get("flights", {}).items():
                lines.append(f"\nFLIGHT: {fnum} — {flight.get('origin','')} → {flight.get('destination','')}")
                lines.append(f"  Departure: {flight.get('scheduled_departure_time_est','')}, Arrival: {flight.get('scheduled_arrival_time_est','')}")
                for date_str, date_info in flight.get("dates", {}).items():
                    status = date_info.get("status", "")
                    if status == "available":
                        seats = date_info.get("available_seats", {})
                        prices = date_info.get("prices", {})
                        lines.append(f"  {date_str}: {status} — seats: be={seats.get('basic_economy',0)}, ec={seats.get('economy',0)}, biz={seats.get('business',0)} | prices: be=${prices.get('basic_economy',0)}, ec=${prices.get('economy',0)}, biz=${prices.get('business',0)}")
                    else:
                        lines.append(f"  {date_str}: {status}")

        return "\n".join(lines)

    def _validate_db_additions(
        self, db_additions: Dict[str, Any], task_dict: Dict[str, Any]
    ) -> List[str]:
        """Structural checks on Phase 2 output. Returns list of errors (empty = valid)."""
        errors = []
        existing_db = self._get_db_state(task_dict)

        if self.domain == "retail":
            # Check orders
            existing_orders = set(existing_db.get("orders", {}).keys())
            for oid, order in db_additions.get("orders", {}).items():
                if oid in existing_orders:
                    errors.append(f"Order {oid} collides with existing order")
                if not order.get("order_id"):
                    errors.append(f"Order {oid} missing order_id")
                if not order.get("user_id"):
                    errors.append(f"Order {oid} missing user_id")
                if not order.get("status"):
                    errors.append(f"Order {oid} missing status")
                valid_statuses = {"pending", "processed", "pending (item modified)", "delivered", "cancelled", "exchange requested", "return requested"}
                if order.get("status") not in valid_statuses:
                    errors.append(f"Order {oid} has invalid status: {order.get('status')}")
                if not order.get("items"):
                    errors.append(f"Order {oid} has no items")
        elif self.domain == "telecom":
            existing_line_ids = {l.get("line_id", "") for l in existing_db.get("lines", [])}
            existing_bill_ids = {b.get("bill_id", "") for b in existing_db.get("bills", [])}
            existing_device_ids = {d.get("device_id", "") for d in existing_db.get("devices", [])}
            valid_line_statuses = {"Active", "Suspended", "Pending Activation", "Closed"}
            valid_bill_statuses = {"Draft", "Issued", "Awaiting Payment", "Paid", "Overdue", "Disputed"}
            for line in db_additions.get("lines", []):
                lid = line.get("line_id", "")
                if lid in existing_line_ids:
                    errors.append(f"Line {lid} collides with existing line")
                if not lid:
                    errors.append("Line missing line_id")
                if not line.get("phone_number"):
                    errors.append(f"Line {lid} missing phone_number")
                if not line.get("plan_id"):
                    errors.append(f"Line {lid} missing plan_id")
                if line.get("status") not in valid_line_statuses:
                    errors.append(f"Line {lid} has invalid status: {line.get('status')}")
            for bill in db_additions.get("bills", []):
                bid = bill.get("bill_id", "")
                if bid in existing_bill_ids:
                    errors.append(f"Bill {bid} collides with existing bill")
                if not bid:
                    errors.append("Bill missing bill_id")
                if not bill.get("customer_id"):
                    errors.append(f"Bill {bid} missing customer_id")
                if bill.get("status") not in valid_bill_statuses:
                    errors.append(f"Bill {bid} has invalid status: {bill.get('status')}")
                if bill.get("total_due") is None:
                    errors.append(f"Bill {bid} missing total_due")
            for device in db_additions.get("devices", []):
                did = device.get("device_id", "")
                if did in existing_device_ids:
                    errors.append(f"Device {did} collides with existing device")
                if not did:
                    errors.append("Device missing device_id")
                valid_device_types = {"phone", "router", "tablet", "watch", "other"}
                if device.get("device_type") not in valid_device_types:
                    errors.append(f"Device {did} has invalid device_type: {device.get('device_type')}")
        else:
            # Airline: Check flights
            existing_flights = set(existing_db.get("flights", {}).keys())
            for fnum, flight in db_additions.get("flights", {}).items():
                if fnum in existing_flights:
                    errors.append(f"Flight {fnum} collides with existing flight")
                if not flight.get("flight_number"):
                    errors.append(f"Flight {fnum} missing flight_number")
                if not flight.get("origin") or not flight.get("destination"):
                    errors.append(f"Flight {fnum} missing origin or destination")
                if not flight.get("scheduled_departure_time_est") or not flight.get("scheduled_arrival_time_est"):
                    errors.append(f"Flight {fnum} missing departure or arrival time")
                if not flight.get("dates"):
                    errors.append(f"Flight {fnum} has no dates")
                for date_str, date_info in flight.get("dates", {}).items():
                    for cabin, price in date_info.get("prices", {}).items():
                        if not isinstance(price, int):
                            errors.append(f"Flight {fnum} date {date_str} {cabin} price is {type(price).__name__}, must be int")

            # Check reservations
            existing_reservations = set(existing_db.get("reservations", {}).keys())
            for rid, res in db_additions.get("reservations", {}).items():
                if rid in existing_reservations:
                    errors.append(f"Reservation {rid} collides with existing reservation")
                if not res.get("reservation_id"):
                    errors.append(f"Reservation {rid} missing reservation_id")
                if not res.get("user_id"):
                    errors.append(f"Reservation {rid} missing user_id")
                if not res.get("flights"):
                    errors.append(f"Reservation {rid} has no flights")

        return errors

    def _escape_braces(self, text: str) -> str:
        """Escape curly braces in text for use with str.format()."""
        return text.replace("{", "{{").replace("}", "}}")

    def _call_db_trap_prompt(
        self,
        decoy_specs: List[Any],
        existing_db_summary: str,
        empty_defaults: Dict[str, Any],
        task_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Render the unified ``db_trap_construction`` prompt, call the LLM,
        validate the structural shape of the response, and retry once on errors.

        ``empty_defaults`` is the dict of top-level collections the prompt is
        expected to return (e.g. ``{"flights": {}, "reservations": {}}`` for
        airline) -- used both as the post-call ``setdefault`` baseline and to
        keep the output shape consistent.
        """
        template = self.prompt_manager.get_prompt("db_trap_construction")
        prompt = template.format(
            decoy_specs=self._escape_braces(json.dumps(decoy_specs, indent=2)),
            db_schema=self._escape_braces(self.db_schema_str),
            existing_db_summary=self._escape_braces(existing_db_summary),
        )

        response = self.caller.call(prompt=prompt, stage_name="db_trap_construction")
        db_additions = LLMResponseParser.extract_json(response)
        for k, v in empty_defaults.items():
            db_additions.setdefault(k, v)

        errors = self._validate_db_additions(db_additions, task_dict)
        if errors:
            print(f"  Phase 2 validation errors: {errors}")
            retry_prompt = prompt + (
                "\n\nYour previous output had structural errors:\n"
                + "\n".join(f"- {e}" for e in errors)
                + "\n\nPlease fix these issues and return corrected JSON."
            )
            response = self.caller.call(
                prompt=retry_prompt, stage_name="db_trap_construction_retry"
            )
            db_additions = LLMResponseParser.extract_json(response)
            for k, v in empty_defaults.items():
                db_additions.setdefault(k, v)

        return db_additions

    # ------------------------------------------------------------------
    # Phase methods (stubs — implemented in later tasks)
    # ------------------------------------------------------------------

    def phase1_strategy(self, task_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 1: Analyze golden actions + DB, produce adversarial plan."""
        actions = task_dict.get("evaluation_criteria", {}).get("actions", [])
        db_state = self._get_db_state(task_dict)

        # Format tool types as readable list
        type_lines = []
        for name, atype in sorted(self.action_types.items()):
            type_lines.append(f"  {name}: {atype}")
        tool_types_str = "\n".join(type_lines)

        strategy_template = self.prompt_manager.get_prompt("adversarial_strategy")
        prompt = strategy_template.format(
            action_sequence_with_types=self._escape_braces(
                self._format_actions_with_types(actions)
            ),
            db_state=self._escape_braces(json.dumps(db_state, indent=2)),
            policy=self._escape_braces(self.policy),
            tool_types=self._escape_braces(tool_types_str),
        )

        response = self.caller.call(prompt=prompt, stage_name="adversarial_strategy")
        return LLMResponseParser.extract_json(response)

    def phase2_db_traps(
        self, strategy: Dict[str, Any], task_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Phase 2: Build decoy entities from strategy. May return empty dicts."""
        decoy_specs = strategy.get("decoy_entities_needed", [])

        if self.domain == "retail":
            return self._phase2_retail(decoy_specs, task_dict)
        elif self.domain == "telecom":
            return self._phase2_telecom(decoy_specs, task_dict)
        else:
            return self._phase2_airline(decoy_specs, task_dict)

    def _phase2_airline(
        self, decoy_specs: List[Any], task_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Airline Phase 2: build flight/reservation decoys."""
        if not decoy_specs:
            return {"flights": {}, "reservations": {}}

        db = self._get_db_state(task_dict)

        existing_flight_ids = sorted(db.get("flights", {}).keys())
        existing_reservation_ids = sorted(db.get("reservations", {}).keys())
        user_id = next(iter(db.get("users", {})), "")

        lines: List[str] = []
        lines.append(f"Existing flight numbers: {', '.join(existing_flight_ids)}")
        lines.append(f"Existing reservation IDs: {', '.join(existing_reservation_ids)}")
        lines.append(f"Owner user_id: {user_id}")
        lines.append("")
        for fnum, flight in db.get("flights", {}).items():
            lines.append(
                f"Flight {fnum}: {flight.get('origin','')}->{flight.get('destination','')} "
                f"dep={flight.get('scheduled_departure_time_est','')} arr={flight.get('scheduled_arrival_time_est','')}"
            )
            for date_str, di in flight.get("dates", {}).items():
                if di.get("status") == "available":
                    prices = di.get("prices", {})
                    seats = di.get("available_seats", {})
                    lines.append(
                        f"  {date_str}: prices(be={prices.get('basic_economy','')}, "
                        f"ec={prices.get('economy','')}, biz={prices.get('business','')}) "
                        f"seats(be={seats.get('basic_economy','')}, "
                        f"ec={seats.get('economy','')}, biz={seats.get('business','')})"
                    )
        for rid, res in db.get("reservations", {}).items():
            lines.append(
                f"Reservation {rid}: {res.get('origin','')}->{res.get('destination','')} "
                f"cabin={res.get('cabin','')} insurance={res.get('insurance','')} "
                f"created={res.get('created_at','')}"
            )

        return self._call_db_trap_prompt(
            decoy_specs=decoy_specs,
            existing_db_summary="\n".join(lines),
            empty_defaults={"flights": {}, "reservations": {}},
            task_dict=task_dict,
        )

    def _phase2_retail(
        self, decoy_specs: List[Any], task_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Retail Phase 2: build decoy order entities."""
        if not decoy_specs:
            return {"orders": {}}

        db = self._get_db_state(task_dict)

        existing_order_ids = sorted(db.get("orders", {}).keys())
        user_id = next(iter(db.get("users", {})), "")

        lines: List[str] = []
        lines.append(f"Existing order IDs: {', '.join(existing_order_ids)}")
        lines.append(f"Owner user_id: {user_id}")
        lines.append("")
        for uid, user in db.get("users", {}).items():
            for pid, pm in user.get("payment_methods", {}).items():
                source = pm.get("source", "")
                if source == "credit_card":
                    lines.append(
                        f"PaymentMethod {pid}: credit_card ({pm.get('brand','')}, last four {pm.get('last_four','')})"
                    )
                elif source == "gift_card":
                    lines.append(
                        f"PaymentMethod {pid}: gift_card (balance=${pm.get('balance', 0)})"
                    )
                elif source == "paypal":
                    lines.append(f"PaymentMethod {pid}: paypal")
            addr = user.get("address", {})
            lines.append(
                f"UserAddress: {addr.get('address1','')} {addr.get('city','')} {addr.get('state','')} {addr.get('zip','')}"
            )
        for oid, order in db.get("orders", {}).items():
            lines.append(
                f"Order {oid}: status={order.get('status','')} items={[i.get('name','') for i in order.get('items',[])]}"
            )
        for pid, product in db.get("products", {}).items():
            lines.append(f"Product {pid}: {product.get('name','')}")
            for iid, variant in product.get("variants", {}).items():
                opts = ", ".join(f"{k}={v}" for k, v in variant.get("options", {}).items())
                lines.append(
                    f"  Variant {iid}: {opts}, price=${variant.get('price','')}, available={variant.get('available','')}"
                )

        return self._call_db_trap_prompt(
            decoy_specs=decoy_specs,
            existing_db_summary="\n".join(lines),
            empty_defaults={"orders": {}},
            task_dict=task_dict,
        )

    def _phase2_telecom(
        self, decoy_specs: List[Any], task_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Telecom Phase 2: build decoy line, bill, and device entities."""
        if not decoy_specs:
            return {"lines": [], "bills": [], "devices": []}

        db = self._get_db_state(task_dict)

        existing_line_ids = sorted(l.get("line_id", "") for l in db.get("lines", []))
        existing_bill_ids = sorted(b.get("bill_id", "") for b in db.get("bills", []))
        existing_device_ids = sorted(d.get("device_id", "") for d in db.get("devices", []))
        customer_id = next(
            (c.get("customer_id", "") for c in db.get("customers", [])), ""
        )

        lines: List[str] = []
        lines.append(f"Existing line IDs: {', '.join(existing_line_ids)}")
        lines.append(f"Existing bill IDs: {', '.join(existing_bill_ids)}")
        lines.append(f"Existing device IDs: {', '.join(existing_device_ids)}")
        lines.append(f"Owner customer_id: {customer_id}")
        lines.append("")
        for plan in db.get("plans", []):
            lines.append(
                f"Plan {plan.get('plan_id','')}: {plan.get('name','')} "
                f"data={plan.get('data_limit_gb','')}GB "
                f"price=${plan.get('price_per_month','')} "
                f"refuel=${plan.get('data_refueling_price_per_gb','')}/GB"
            )
        for line in db.get("lines", []):
            lines.append(
                f"Line {line.get('line_id','')}: phone={line.get('phone_number','')} "
                f"status={line.get('status','')} plan={line.get('plan_id','')} "
                f"device={line.get('device_id','')} "
                f"data_used={line.get('data_used_gb','')}GB "
                f"contract_end={line.get('contract_end_date','')} "
                f"suspension_start={line.get('suspension_start_date','')}"
            )
        for bill in db.get("bills", []):
            lines.append(
                f"Bill {bill.get('bill_id','')}: status={bill.get('status','')} "
                f"total_due=${bill.get('total_due','')} "
                f"period={bill.get('period_start','')} to {bill.get('period_end','')} "
                f"due={bill.get('due_date','')}"
            )
        for device in db.get("devices", []):
            lines.append(
                f"Device {device.get('device_id','')}: type={device.get('device_type','')} "
                f"model={device.get('model','')} "
                f"esim={device.get('is_esim_capable','')} "
                f"activated={device.get('activated','')}"
            )

        return self._call_db_trap_prompt(
            decoy_specs=decoy_specs,
            existing_db_summary="\n".join(lines),
            empty_defaults={"lines": [], "bills": [], "devices": []},
            task_dict=task_dict,
        )

    def phase3_scenario(
        self,
        strategy: Dict[str, Any],
        db_additions: Dict[str, Any],
        task_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Phase 3: Write adversarial user scenario from strategy + traps."""
        actions = task_dict.get("evaluation_criteria", {}).get("actions", [])
        instr = task_dict.get("user_scenario", {}).get("instructions", {})

        original_instructions = json.dumps(
            {
                "reason_for_call": instr.get("reason_for_call", ""),
                "known_info": instr.get("known_info", ""),
                "task_instructions": instr.get("task_instructions", ""),
            },
            indent=2,
        )

        # Format DB additions for prompt (or "None" if empty)
        if self.domain == "retail":
            has_additions = bool(db_additions.get("orders"))
        elif self.domain == "telecom":
            has_additions = any(db_additions.get(k) for k in ("lines", "bills", "devices"))
        else:
            has_additions = any(db_additions.get(k) for k in ("flights", "reservations"))
        db_additions_str = (
            json.dumps(db_additions, indent=2) if has_additions else "None — no decoy entities were added."
        )

        scenario_template = self.prompt_manager.get_prompt("adversarial_scenario")
        prompt = scenario_template.format(
            adversarial_strategy=self._escape_braces(json.dumps(strategy, indent=2)),
            db_additions=self._escape_braces(db_additions_str),
            action_sequence=self._escape_braces(
                self._format_actions_with_types(actions)
            ),
            db_summary=self._escape_braces(self._extract_db_summary(task_dict)),
            policy=self._escape_braces(self.policy),
            original_instructions=self._escape_braces(original_instructions),
        )

        response = self.caller.call(prompt=prompt, stage_name="adversarial_scenario")
        return LLMResponseParser.extract_json(response)

    def phase3_scenario_lite(self, task_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Lite Phase 3: Write a mildly adversarial scenario without strategy/traps."""
        actions = task_dict.get("evaluation_criteria", {}).get("actions", [])
        instr = task_dict.get("user_scenario", {}).get("instructions", {})

        original_instructions = json.dumps(
            {
                "reason_for_call": instr.get("reason_for_call", ""),
                "known_info": instr.get("known_info", ""),
                "task_instructions": instr.get("task_instructions", ""),
            },
            indent=2,
        )

        lite_template = self.prompt_manager.get_prompt("adversarial_scenario_lite")
        prompt = lite_template.format(
            action_sequence=self._escape_braces(
                self._format_actions_with_types(actions)
            ),
            db_summary=self._escape_braces(self._extract_db_summary(task_dict)),
            policy=self._escape_braces(self.policy),
            original_instructions=self._escape_braces(original_instructions),
        )

        response = self.caller.call(prompt=prompt, stage_name="adversarial_scenario_lite")
        return LLMResponseParser.extract_json(response)

    def evolve_task(self, task_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Full pipeline: 3 phases, returns assembled result.

        Returns dict with keys:
            reason_for_call, known_info, task_instructions,
            db_additions, adversarial_strategy
        """
        strategy = self.phase1_strategy(task_dict)
        db_additions = self.phase2_db_traps(strategy, task_dict)
        scenario = self.phase3_scenario(strategy, db_additions, task_dict)

        return {
            "reason_for_call": scenario.get("reason_for_call", ""),
            "known_info": scenario.get("known_info", ""),
            "task_instructions": scenario.get("task_instructions", ""),
            "db_additions": db_additions,
            "adversarial_strategy": strategy,
        }
