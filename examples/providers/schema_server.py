"""An MCP server whose tool schemas use JSON Schema features some providers reject (Gemini in particular):
nested models ($defs/$ref), Optional fields (anyOf with null), single-value Literals (const), defaults and
URI formats. Used by examples/providers/suite.yaml to show that sanitising keeps these tools usable.

python schema_server.py
"""

from typing import Literal

from pydantic import BaseModel, Field

try:  # mcp 2.x
    from mcp.server.mcpserver import MCPServer as Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as Server


class Dimensions(BaseModel):
    length_cm: float
    width_cm: float
    height_cm: float | None = None  # anyOf [number, null]


class Item(BaseModel):
    sku: str
    name: str
    dimensions: Dimensions  # nested model -> $defs + $ref


def build() -> Server:
    s = Server("schema")

    @s.tool()
    def add_catalogue_item(
        item: Item,
        source: Literal["api"] = "api",  # const
        datasheet_url: str | None = Field(default=None, json_schema_extra={"format": "uri"}),
    ) -> str:
        """Add a product to the warehouse catalogue (SKU, name and package dimensions)."""
        d = item.dimensions
        return (
            f"catalogue entry CAT-88 created: {item.sku} {item.name}, {d.length_cm}x{d.width_cm} cm"
        )

    return s


if __name__ == "__main__":
    build().run()
