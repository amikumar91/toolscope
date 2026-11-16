"""Two small demo MCP servers with deliberately confusable tools (decoys), so selection metrics have teeth.

    python servers.py shop   # orders, shipments, refunds
    python servers.py crm    # customers, tickets
"""
import sys

try:  # mcp 2.x
    from mcp.server.mcpserver import MCPServer as Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as Server

ORDERS = {
    "ORD-1042": {"status": "shipped", "carrier": "BlueDart", "tracking": "BD778812", "total": 2499, "customer": "CUS-7"},
    "ORD-1043": {"status": "processing", "carrier": None, "tracking": None, "total": 899, "customer": "CUS-7"},
    "ORD-2001": {"status": "delivered", "carrier": "Delhivery", "tracking": "DL11002", "total": 15999, "customer": "CUS-9"},
}
CUSTOMERS = {
    "CUS-7": {"name": "Priya Raman", "email": "priya@example.com", "tier": "gold"},
    "CUS-9": {"name": "Arjun Mehta", "email": "arjun@example.com", "tier": "silver"},
}


def shop() -> Server:
    s = Server("shop")

    @s.tool()
    def get_order_status(order_id: str) -> str:
        """Get the current fulfilment status of an order (processing, shipped, delivered)."""
        o = ORDERS.get(order_id)
        if not o:
            raise ValueError(f"order {order_id} not found")
        return f"{order_id} status: {o['status']}"

    @s.tool()
    def get_order_details(order_id: str) -> str:
        """Get line items, total amount and customer for an order. Does not include shipping status."""
        o = ORDERS.get(order_id)
        if not o:
            raise ValueError(f"order {order_id} not found")
        return f"{order_id}: total INR {o['total']}, customer {o['customer']}"

    @s.tool()
    def track_shipment(tracking_number: str) -> str:
        """Track a parcel with the carrier using its tracking number (not the order id)."""
        for o in ORDERS.values():
            if o["tracking"] == tracking_number:
                return f"{tracking_number} with {o['carrier']}: out for delivery"
        raise ValueError(f"unknown tracking number {tracking_number}")

    @s.tool()
    def cancel_order(order_id: str, reason: str) -> str:
        """Cancel an order. Irreversible. Only for orders that have not shipped."""
        o = ORDERS.get(order_id)
        if not o:
            raise ValueError(f"order {order_id} not found")
        if o["status"] != "processing":
            raise ValueError(f"cannot cancel: order is {o['status']}")
        return f"{order_id} cancelled ({reason})"

    @s.tool()
    def issue_refund(order_id: str, amount: float) -> str:
        """Refund part or all of a delivered order's amount to the customer."""
        o = ORDERS.get(order_id)
        if not o:
            raise ValueError(f"order {order_id} not found")
        if amount > o["total"]:
            raise ValueError("refund exceeds order total")
        return f"refund of INR {amount:g} issued for {order_id}"

    @s.tool()
    def list_orders(customer_id: str) -> str:
        """List all order ids placed by a customer id."""
        ids = [k for k, v in ORDERS.items() if v["customer"] == customer_id]
        return f"orders for {customer_id}: {', '.join(ids) or 'none'}"

    return s


def crm() -> Server:
    s = Server("crm")

    @s.tool()
    def find_customer_by_email(email: str) -> str:
        """Look up a customer record (id, name, tier) by email address."""
        for cid, c in CUSTOMERS.items():
            if c["email"] == email:
                return f"{cid}: {c['name']}, tier {c['tier']}"
        raise ValueError(f"no customer with email {email}")

    @s.tool()
    def get_customer(customer_id: str) -> str:
        """Get a customer's profile (name, email, loyalty tier) by customer id."""
        c = CUSTOMERS.get(customer_id)
        if not c:
            raise ValueError(f"customer {customer_id} not found")
        return f"{customer_id}: {c['name']} <{c['email']}>, tier {c['tier']}"

    @s.tool()
    def create_ticket(customer_id: str, subject: str, priority: str = "normal") -> str:
        """Open a support ticket for a customer. priority: low | normal | high."""
        if priority not in ("low", "normal", "high"):
            raise ValueError("priority must be low, normal or high")
        return f"ticket TCK-501 opened for {customer_id} [{priority}]: {subject}"

    @s.tool()
    def get_customer_orders_summary(customer_id: str) -> str:
        """Lifetime spend and order count for a customer (aggregate only, no order ids)."""
        os_ = [v for v in ORDERS.values() if v["customer"] == customer_id]
        return f"{customer_id}: {len(os_)} orders, lifetime INR {sum(o['total'] for o in os_)}"

    return s


if __name__ == "__main__":
    {"shop": shop, "crm": crm}[sys.argv[1]]().run()
