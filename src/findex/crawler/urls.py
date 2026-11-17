"""URL normalization and explicit crawl scope."""

from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit


def canonical_url(url: str, base: str | None = None) -> str | None:
    """Normalize HTTP URLs for deduplication; reject credentials and other schemes."""
    if base is not None:
        url = urljoin(base, url)
    try:
        parts = urlsplit(url)
        host = parts.hostname
        port = parts.port
    except ValueError:
        return None
    scheme = parts.scheme.lower()
    if scheme not in ("http", "https") or not host or parts.username or parts.password:
        return None
    host = host.lower().rstrip(".")
    if ":" in host:
        host = f"[{host}]"
    default_port = 80 if scheme == "http" else 443
    authority = host if port in (None, default_port) else f"{host}:{port}"
    path = parts.path.rstrip("/") or "/"
    query = urlencode(sorted(parse_qsl(parts.query, keep_blank_values=True)))
    return urlunsplit((scheme, authority, path, query, ""))


def origin(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}"


def in_scope(url: str, domains: set[str]) -> bool:
    host = urlsplit(url).hostname
    return host is not None and any(
        host == domain or host.endswith(f".{domain}") for domain in domains
    )
