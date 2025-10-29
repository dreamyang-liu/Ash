from typing import Annotated, Any

from ddgs import DDGS
from fastmcp import FastMCP
from pydantic import Field

mcp = FastMCP("ddgs-mcp")


@mcp.tool()
async def search(
    query: Annotated[str, Field(description="Search query string. Supports operators like 'cats dogs' (OR), '\"cats and dogs\"' (exact phrase), 'cats -dogs' (exclude), 'cats +dogs' (include), 'cats filetype:pdf' (file type), 'site:example.com' (specific site), 'intitle:dogs' (in title), 'inurl:cats' (in URL)")],
    max_results: Annotated[int, Field(ge=1, description="Max results to return (default: 10). Set to None for all unique results")] = 10,
    page: Annotated[int, Field(ge=1, description="1-indexed page for pagination. Each page contains results from selected backends")] = 1,
    region: Annotated[str, Field(description="Region code format: {country}-{language}. Examples: us-en, uk-en, ru-ru, cn-zh, it-it, ar-es, pl-pl. Strongly influences search quality")] = "us-en",
    safesearch: Annotated[str, Field(description="Safe search filter: 'on' (strict), 'moderate' (balanced), 'off' (disabled)")] = "moderate",
    timelimit: Annotated[str | None, Field(description="Time limit for results: 'd' (day), 'w' (week), 'm' (month), 'y' (year)")] = None,
    backend: Annotated[str, Field(description="Backend engines (comma-separated or single). Available: bing, brave, duckduckgo, google, mojeek, mullvad_brave, mullvad_google, yandex, yahoo, wikipedia. Use 'auto' for automatic selection with fallback. Example: 'google,brave'")] = "auto",
) -> list[dict[str, Any]]:
    """Web search tool with configurable parameters and advanced search operators.

    Example queries:
    - Basic search: "neurophysiology flickering light"
    - Exact phrase: "machine learning algorithms"
    - File type: "climate change filetype:pdf"
    - Site specific: "python tutorials site:github.com"
    - Exclude terms: "cats -dogs"
    - Include terms: "programming +python"

    Returns list of dictionaries with 'title', 'href', and 'body' fields.
    """
    return DDGS().text(
        query=query,
        region=region,
        safesearch=safesearch,
        timelimit=timelimit,
        max_results=max_results,
        page=page,
        backend=backend,
    )


@mcp.tool()
async def search_images(
    query: Annotated[str, Field(description="Image search query. Use descriptive terms for better results")],
    max_results: Annotated[int, Field(ge=1, description="Max results to return (default: 10)")] = 10,
    page: Annotated[int, Field(ge=1, description="Page number for pagination")] = 1,
    region: Annotated[str, Field(description="Region code (format: country-language). Examples: us-en, uk-en, cn-zh")] = "us-en",
    safesearch: Annotated[str, Field(description="Safe search: 'on', 'moderate', 'off'")] = "moderate",
    timelimit: Annotated[str | None, Field(description="Time limit: 'd' (day), 'w' (week), 'm' (month), 'y' (year)")] = None,
    backend: Annotated[str, Field(description="Backend engine. Available: duckduckgo (only option for images)")] = "auto",
    size: Annotated[str | None, Field(description="Image size filter: 'Small', 'Medium', 'Large', 'Wallpaper'")] = None,
    color: Annotated[str | None, Field(description="Color filter: 'color', 'Monochrome', 'Red', 'Orange', 'Yellow', 'Green', 'Blue', 'Purple', 'Pink', 'Brown', 'Black', 'Gray', 'Teal', 'White'")] = None,
    type_image: Annotated[str | None, Field(description="Image type: 'photo', 'clipart', 'gif', 'transparent', 'line'")] = None,
    layout: Annotated[str | None, Field(description="Image layout/orientation: 'Square', 'Tall', 'Wide'")] = None,
    license_image: Annotated[str | None, Field(description="License type: 'any' (All Creative Commons), 'Public' (Public Domain), 'Share' (Free to Share), 'ShareCommercially' (Free Commercial Share), 'Modify' (Free to Modify/Share), 'ModifyCommercially' (Free Commercial Modify)")] = None,
) -> list[dict[str, Any]]:
    """Image search tool with advanced filtering options.

    Example usage:
    - Basic: query="butterfly", color="Monochrome"
    - Specific: query="sunset landscape", size="Wallpaper", layout="Wide"
    - Licensed: query="business photos", license_image="ShareCommercially"

    Returns list with 'title', 'image', 'thumbnail', 'url', 'height', 'width', 'source' fields.
    """
    return DDGS().images(
        query=query,
        region=region,
        safesearch=safesearch,
        timelimit=timelimit,
        max_results=max_results,
        page=page,
        backend=backend,
        size=size,
        color=color,
        type_image=type_image,
        layout=layout,
        license_image=license_image,
    )


@mcp.tool()
async def search_videos(
    query: Annotated[str, Field(description="Video search query. Use descriptive terms for better results")],
    max_results: Annotated[int, Field(ge=1, description="Max results to return (default: 10)")] = 10,
    page: Annotated[int, Field(ge=1, description="Page number for pagination")] = 1,
    region: Annotated[str, Field(description="Region code (format: country-language). Examples: us-en, uk-en, cn-zh")] = "us-en",
    safesearch: Annotated[str, Field(description="Safe search: 'on', 'moderate', 'off'")] = "moderate",
    timelimit: Annotated[str | None, Field(description="Time limit: 'd' (day), 'w' (week), 'm' (month). Note: 'y' (year) not available for videos")] = None,
    backend: Annotated[str, Field(description="Backend engine. Available: duckduckgo (only option for videos)")] = "auto",
    resolution: Annotated[str | None, Field(description="Video resolution filter: 'high' (HD quality), 'standard' (SD quality)")] = None,
    duration: Annotated[str | None, Field(description="Video duration filter: 'short' (<4min), 'medium' (4-20min), 'long' (>20min)")] = None,
    license_videos: Annotated[str | None, Field(description="Video license filter: 'creativeCommon' (Creative Commons), 'youtube' (YouTube license)")] = None,
) -> list[dict[str, Any]]:
    """Video search tool with duration, resolution, and license filtering.

    Example usage:
    - Basic: query="cooking tutorial", duration="medium"
    - High quality: query="nature documentary", resolution="high", duration="long"
    - Licensed: query="music videos", license_videos="creativeCommon"

    Returns list with 'content', 'description', 'duration', 'title', 'uploader', 'published' fields.
    """
    return DDGS().videos(
        query=query,
        region=region,
        safesearch=safesearch,
        timelimit=timelimit,
        max_results=max_results,
        page=page,
        backend=backend,
        resolution=resolution,
        duration=duration,
        license_videos=license_videos,
    )


@mcp.tool()
async def search_news(
    query: Annotated[str, Field(description="News search query. Use current events, names, topics for better results")],
    max_results: Annotated[int, Field(ge=1, description="Max results to return (default: 10)")] = 10,
    page: Annotated[int, Field(ge=1, description="Page number for pagination")] = 1,
    region: Annotated[str, Field(description="Region code (format: country-language). Examples: us-en, uk-en, it-it. Affects news sources and relevance")] = "us-en",
    safesearch: Annotated[str, Field(description="Safe search: 'on', 'moderate', 'off'")] = "moderate",
    timelimit: Annotated[str | None, Field(description="Time limit: 'd' (day), 'w' (week), 'm' (month). Note: 'y' (year) not available for news")] = None,
    backend: Annotated[str, Field(description="Backend engines (comma-separated). Available: bing, duckduckgo, yahoo. Use 'auto' for automatic selection with fallback")] = "auto",
) -> list[dict[str, Any]]:
    """News search tool for current events and recent articles.

    Example usage:
    - Current events: query="climate summit", timelimit="d", region="us-en"
    - Local news: query="etna eruption", region="it-it"
    - Recent news: query="technology updates", timelimit="w"

    Returns list with 'date', 'title', 'body', 'url', 'image', 'source' fields.
    """
    return DDGS().news(
        query=query,
        region=region,
        safesearch=safesearch,
        timelimit=timelimit,
        max_results=max_results,
        page=page,
        backend=backend,
    )


@mcp.tool()
async def search_books(
    query: Annotated[str, Field(description="Book search query. Use book titles, author names, subjects, or ISBN for better results")],
    max_results: Annotated[int, Field(ge=1, description="Max results to return (default: 10)")] = 10,
    page: Annotated[int, Field(ge=1, description="Page number for pagination")] = 1,
    backend: Annotated[str, Field(description="Backend engine. Available: annasarchive (only option for books). Searches Anna's Archive digital library")] = "auto",
) -> list[dict[str, Any]]:
    """Book search tool using Anna's Archive digital library.

    Example usage:
    - By title and author: query="sea wolf jack london"
    - By subject: query="machine learning textbook"
    - By author: query="dolphins cousteau"
    - Technical books: query="python programming guide"

    Returns list with 'title', 'author', 'publisher', 'info', 'url', 'thumbnail' fields.
    Info field contains language, format (epub/pdf), file size, and book type.
    """
    return DDGS().books(
        query=query,
        max_results=max_results,
        page=page,
        backend=backend,
    )


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=3000)
