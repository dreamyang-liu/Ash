"""Generate golden parity vectors from the official Python ddgs (v9.14.4).

Output: /opt/dlami/nvme/projects/ddgs-go/testdata/parity_golden.json
Consumed by parity_test.go in ddgs-go to assert 1:1 behavior.
"""
import json
import sys

sys.path.insert(0, "/tmp/ddgs-src")

from ddgs.utils import _normalize_text, _normalize_url, _normalize_date, _extract_vqd
from ddgs.similarity import SimpleFilterRanker
from ddgs.results import ResultsAggregator, TextResult
from ddgs.exceptions import DDGSException

golden = {}

# --- 1. normalize_text ---
text_cases = [
    "<b>hello</b> world",
    "a &amp; b &lt;c&gt; &quot;d&quot; &#39;e&#39;",
    "  a \t b \n c  ",
    "line1\nline2\r\nline3",
    "zero​width​space",
    "café naïve résumé",
    "éclair",  # combining accent -> NFC
    "tab\there",
    "<div class='x'>nested <span>tags</span></div>",
    "&amp;amp; double escaped",
    "control\x01chars\x02here",
    "ideographic　space",
    "",
    "   ",
    "普通话 中文 测试",
    "emoji 🎉 test",
    "<p>Multi</p><p>Paragraph</p>",
    "a&nbsp;b",
    "quote &ldquo;smart&rdquo; quotes",
    "mixed <b>HTML &amp; entities</b> ​  whitespace",
]
golden["normalize_text"] = [{"in": t, "out": _normalize_text(t)} for t in text_cases]

# --- 2. normalize_url ---
url_cases = [
    "https://x.com/a%20b",
    "https://x.com/a b",
    "https://example.com/path?q=hello%20world&x=%2Fslash",
    "https://en.wikipedia.org/wiki/G%C3%B6del",
    "",
    "https://x.com/%E4%B8%AD%E6%96%87",
    "https://x.com/already+plus",
    "http://a.b/c%2Bd",   # encoded plus
    "https://x.com/double%2520encoded",
]
golden["normalize_url"] = [{"in": u, "out": _normalize_url(u)} for u in url_cases]

# --- 3. normalize_date ---
date_cases_int = [0, 86400, 1700000000, 1234567890]
date_cases_str = ["2024-01-01", "3 days ago", ""]
golden["normalize_date"] = (
    [{"in": d, "int": True, "out": _normalize_date(d)} for d in date_cases_int]
    + [{"in": d, "int": False, "out": _normalize_date(d)} for d in date_cases_str]
)

# --- 4. extract_vqd ---
vqd_cases = [
    'x vqd="4-12345" y',
    'href="?q=x&vqd=4-999&other=1"',
    "vqd='4-abc'",
    'a vqd="first" b vqd="second"',       # first match wins
    'vqd=4-noquote&next',
    "no token",
]
out = []
for c in vqd_cases:
    try:
        out.append({"in": c, "out": _extract_vqd(c.encode(), "q"), "err": False})
    except DDGSException:
        out.append({"in": c, "out": "", "err": True})
golden["extract_vqd"] = out

# --- 5. SimpleFilterRanker ---
ranker = SimpleFilterRanker()
rank_docs = [
    {"title": "unrelated stuff", "href": "https://a.com", "body": "nothing here"},
    {"title": "", "href": "https://b.com", "body": "golang tutorial content"},
    {"title": "golang guide", "href": "https://c.com", "body": "nothing"},
    {"title": "golang docs", "href": "https://d.com", "body": "golang reference"},
    {"title": "Go language", "href": "https://en.wikipedia.org/wiki/Go", "body": "x"},
    {"title": "Category: Golang - Wikimedia Commons", "href": "https://commons.wikimedia.org/x", "body": "y"},
    {"title": "GOLANG UPPERCASE", "href": "https://e.com", "body": "GOLANG BODY"},
    {"title": "short go", "href": "https://f.com", "body": "the go language"},  # 'go' < min_token_length
    {"title": "concurrency", "href": "https://g.com", "body": "with golang"},
    {"title": "description fallback", "href": "https://h.com", "description": "golang described"},
]
rank_queries = ["golang", "golang concurrency", "go", "GOLANG TUTORIAL", "xyz-nonexistent"]
golden["ranker"] = []
for q in rank_queries:
    ranked = ranker.rank([dict(d) for d in rank_docs], q)
    golden["ranker"].append({"query": q, "docs": rank_docs, "order": [d["href"] for d in ranked]})

# --- 6. ResultsAggregator ---
def tr(title, href, body):
    r = TextResult()
    r.title = title; r.href = href; r.body = body
    return r

agg = ResultsAggregator({"href", "image", "url", "embed_url"})
agg_input = [
    ("B", "https://b.com", "short b"),
    ("A", "https://a.com", "short"),
    ("A2", "https://a.com", "a much longer body text here"),
    ("C", "https://c.com", "c body"),
    ("B2", "https://b.com", "b"),          # shorter -> keep original
    ("A3", "https://a.com", "mid body"),   # third occurrence of a.com
]
agg.extend([tr(*x) for x in agg_input])
golden["aggregator"] = {
    "input": [{"title": t, "href": h, "body": b} for t, h, b in agg_input],
    "output": agg.extract_dicts(),
}

with open("/opt/dlami/nvme/projects/ddgs-go/testdata/parity_golden.json", "w") as f:
    json.dump(golden, f, ensure_ascii=False, indent=1)
print("golden vectors written:", {k: len(v) if isinstance(v, list) else 1 for k, v in golden.items()})
