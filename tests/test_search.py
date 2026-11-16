import pytest

from conftest import SHOP_TOOLS, spec
from toolscope.search import SEARCH_TOOL_SPEC, BM25Retriever, _tokens
from toolscope.trace import SEARCH_TOOL


def test_tokenizes_camel_and_snake_case():
    assert _tokens("getOrderStatus") == ["get", "order", "status"]
    assert _tokens("get_order_status") == ["get", "order", "status"]
    assert _tokens("findCustomer_byEmail v2!") == ["find", "customer", "by", "email", "v2"]
    assert _tokens("") == []


@pytest.fixture
def bm25():
    return BM25Retriever(SHOP_TOOLS)


@pytest.mark.parametrize(
    "query,best",
    [
        ("order status", "shop__get_order_status"),
        ("cancel an order", "shop__cancel_order"),
        ("customer email lookup", "crm__find_customer_by_email"),
        ("order total line items", "shop__get_order_details"),
    ],
)
def test_ranking_sanity(bm25, query, best):
    assert bm25.search(query, 3)[0] == best


def test_parameter_names_are_indexed(bm25):
    assert bm25.search("reason", 5) == ["shop__cancel_order"]


def test_empty_or_unmatched_query_returns_nothing(bm25):
    assert bm25.search("", 5) == []
    assert bm25.search("zzz quux", 5) == []


def test_k_bounds(bm25):
    assert bm25.search("order", 0) == []
    assert len(bm25.search("order", 1)) == 1
    hits = bm25.search("order", 100)  # k larger than the catalogue
    assert len(hits) == len(set(hits)) <= len(SHOP_TOOLS)
    assert "crm__find_customer_by_email" not in hits  # zero-score docs are dropped


def test_empty_catalogue():
    assert BM25Retriever([]).search("anything", 5) == []


def test_camel_case_tool_names_are_searchable():
    r = BM25Retriever(
        [spec("gh", "createPullRequest", "Open a PR"), spec("gh", "listIssues", "List issues")]
    )
    assert r.search("create pull request", 2)[0] == "gh__createPullRequest"


def test_search_tool_spec_shape():
    fn = SEARCH_TOOL_SPEC["function"]
    assert fn["name"] == SEARCH_TOOL and fn["parameters"]["required"] == ["query"]
