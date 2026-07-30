"""Engine-level parity golden data.

1. payload parity: build_payload() output per engine with fixed inputs
   (network-dependent/random values monkeypatched to fixed sentinels).
2. fixture parity: live-captured engine responses + Python's extracted dicts.

Output:
  testdata/parity_payloads.json
  testdata/fixtures/*.{html,json}
  testdata/parity_extract.json
"""
import json, os, sys
sys.path.insert(0, "/tmp/ddgs-src")

OUT = "/opt/dlami/nvme/projects/ddgs-go/testdata"
os.makedirs(f"{OUT}/fixtures", exist_ok=True)

from ddgs.engines.duckduckgo import Duckduckgo
from ddgs.engines.duckduckgo_images import DuckduckgoImages
from ddgs.engines.duckduckgo_news import DuckduckgoNews
from ddgs.engines.duckduckgo_videos import DuckduckgoVideos
from ddgs.engines.google import Google
from ddgs.engines.brave import Brave
from ddgs.engines.mojeek import Mojeek
from ddgs.engines.yahoo import Yahoo
from ddgs.engines.yahoo_news import YahooNews
from ddgs.engines.bing_news import BingNews
from ddgs.engines.bing_images import BingImages
from ddgs.engines.annasarchive import AnnasArchive

# --- 1. payload parity -------------------------------------------------
# Monkeypatch network/random bits to sentinels.
DuckduckgoImages._get_vqd = lambda self, q: "VQD_SENTINEL"
DuckduckgoNews._get_vqd = lambda self, q: "VQD_SENTINEL"
DuckduckgoVideos._get_vqd = lambda self, q: "VQD_SENTINEL"

class FakeClient:
    class client:
        @staticmethod
        def set_cookies(url, cookies): pass
        @staticmethod
        def headers_update(h): pass

def mk(cls):
    e = cls.__new__(cls)
    e.http_client = FakeClient()
    return e

CASES = [
    {"query": "golang test", "region": "us-en", "safesearch": "moderate", "timelimit": None, "page": 1},
    {"query": "golang test", "region": "de-de", "safesearch": "on", "timelimit": "w", "page": 2},
    {"query": "golang test", "region": "us-en", "safesearch": "off", "timelimit": "y", "page": 3},
]

ENGINES = {
    "duckduckgo_text": (Duckduckgo, []),
    "duckduckgo_images": (DuckduckgoImages, []),
    "duckduckgo_news": (DuckduckgoNews, []),
    "duckduckgo_videos": (DuckduckgoVideos, []),
    "google": (Google, []),
    "brave": (Brave, []),
    "mojeek": (Mojeek, []),
    "yahoo": (Yahoo, []),
    "yahoo_news": (YahooNews, []),
    "bing_news": (BingNews, ["qft"]),   # qft for d has interval="4" fixed -> keep
    "bing_images": (BingImages, ["qft"]),  # qft depends on timelimit key name (day/week) not d/w -> mark
    "annasarchive": (AnnasArchive, []),
}

payloads = {}
for name, (cls, _ignore) in ENGINES.items():
    e = mk(cls)
    rows = []
    for c in CASES:
        try:
            p = e.build_payload(**c)
            rows.append({"case": c, "payload": {k: str(v) for k, v in p.items()}})
        except Exception as ex:
            rows.append({"case": c, "error": f"{type(ex).__name__}: {ex}"})
    payloads[name] = rows

with open(f"{OUT}/parity_payloads.json", "w") as f:
    json.dump(payloads, f, ensure_ascii=False, indent=1)
print("payloads:", {k: len(v) for k, v in payloads.items()})

# --- 2. fixture parity --------------------------------------------------
from ddgs.http_client import HttpClient

client = HttpClient(timeout=15)
fixtures = {}

def capture(name, method, url, **kw):
    resp = client.request(method, url, **kw)
    if resp.status_code != 200:
        print(f"capture {name}: HTTP {resp.status_code}, skipped"); return None
    path = f"{OUT}/fixtures/{name}"
    with open(path, "wb") as f:
        f.write(resp.content)
    return resp.text

# duckduckgo text html
ddg_html = capture("ddg_text.html", "POST", "https://html.duckduckgo.com/html/",
                   data={"q": "golang concurrency", "b": "", "l": "us-en"})

# real vqd for JSON endpoints
vqd_resp = client.request("GET", "https://duckduckgo.com", params={"q": "golang"})
from ddgs.utils import _extract_vqd
vqd = _extract_vqd(vqd_resp.content, "golang")

ddg_news = capture("ddg_news.json", "GET", "https://duckduckgo.com/news.js",
                   params={"l": "us-en", "o": "json", "noamp": "1", "q": "golang", "vqd": vqd, "p": "-1"})
ddg_videos = capture("ddg_videos.json", "GET", "https://duckduckgo.com/v.js",
                     params={"l": "us-en", "o": "json", "q": "golang", "vqd": vqd, "f": ",,,", "p": "-1"})

extract_golden = {}

if ddg_html:
    e = Duckduckgo.__new__(Duckduckgo)
    results = e.extract_results(ddg_html)
    results = e.post_extract_results(results)
    extract_golden["ddg_text"] = [r.__dict__ for r in results]

if ddg_news:
    e = DuckduckgoNews.__new__(DuckduckgoNews)
    extract_golden["ddg_news"] = [r.__dict__ for r in e.extract_results(ddg_news)]

if ddg_videos:
    e = DuckduckgoVideos.__new__(DuckduckgoVideos)
    extract_golden["ddg_videos"] = [r.__dict__ for r in e.extract_results(ddg_videos)]

with open(f"{OUT}/parity_extract.json", "w") as f:
    json.dump(extract_golden, f, ensure_ascii=False, indent=1)
print("extract golden:", {k: len(v) for k, v in extract_golden.items()})
