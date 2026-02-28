from __future__ import annotations

import os
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List


def _safe_read_text(path: str, max_bytes: int = 120_000) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            return handle.read(max_bytes)
    except Exception:
        return ""


class CodeIntelToolkit:
    def __init__(self, workspace_root: str | None = None) -> None:
        self.workspace_root = os.path.abspath(workspace_root or os.getcwd())
        self.max_files = int(os.environ.get("EVOAI_CODE_INTEL_MAX_FILES", "500"))
        self.max_matches = int(os.environ.get("EVOAI_CODE_INTEL_MAX_MATCHES", "8"))

    def _iter_python_files(self) -> List[str]:
        results: List[str] = []
        for root, dirs, files in os.walk(self.workspace_root):
            # Exclude venvs, git, caches, and site-packages from code analysis
            dirs[:] = [d for d in dirs if not d.startswith(".venv") and d not in {".git", "__pycache__", "v4env", "site-packages", "dist", "build"}]
            for name in files:
                if not name.endswith(".py"):
                    continue
                results.append(os.path.join(root, name))
                if len(results) >= self.max_files:
                    return results
        return results

    def _extract_query_tokens(self, query: str) -> List[str]:
        tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", str(query or ""))
        seen = set()
        clean: List[str] = []
        for token in tokens:
            lower = token.lower()
            if lower in seen:
                continue
            seen.add(lower)
            clean.append(token)
        return clean[:10]

    def analyze(self, query: str) -> Dict[str, Any]:
        files = self._iter_python_files()
        tokens = self._extract_query_tokens(query)

        module_count = 0
        function_count = 0
        class_count = 0
        top_large: List[tuple[str, int]] = []
        matches: List[Dict[str, Any]] = []

        for path in files:
            content = _safe_read_text(path)
            if not content:
                continue
            module_count += 1
            function_count += len(re.findall(r"^\s*def\s+", content, flags=re.MULTILINE))
            class_count += len(re.findall(r"^\s*class\s+", content, flags=re.MULTILINE))
            line_count = content.count("\n") + 1
            top_large.append((path, line_count))

            if tokens and len(matches) < self.max_matches:
                lines = content.splitlines()
                for idx, line in enumerate(lines, start=1):
                    low = line.lower()
                    hit = next((tok for tok in tokens if tok.lower() in low), None)
                    if not hit:
                        continue
                    matches.append(
                        {
                            "token": hit,
                            "file": os.path.relpath(path, self.workspace_root),
                            "line": idx,
                            "snippet": line.strip()[:200],
                        }
                    )
                    if len(matches) >= self.max_matches:
                        break

        top_large_sorted = sorted(top_large, key=lambda item: item[1], reverse=True)[:3]
        hotspots = [f"{os.path.relpath(path, self.workspace_root)}:{count} lines" for path, count in top_large_sorted]

        if matches:
            first = matches[0]
            summary = (
                f"Code intel: scanned {module_count} modules with {function_count} functions and {class_count} classes. "
                f"Top match '{first['token']}' at {first['file']}:{first['line']}. "
                f"Hotspots: {', '.join(hotspots) if hotspots else 'none'}"
            )
        else:
            summary = (
                f"Code intel: scanned {module_count} modules with {function_count} functions and {class_count} classes. "
                f"No direct token matches were found for this query. "
                f"Hotspots: {', '.join(hotspots) if hotspots else 'none'}"
            )

        return {
            "summary": summary,
            "matches": matches,
            "hotspots": hotspots,
            "module_count": module_count,
            "function_count": function_count,
            "class_count": class_count,
        }


class ResearchToolkit:
    URL_RE = re.compile(r"https?://[^\s)\]]+")

    def __init__(self) -> None:
        self.allow_web = os.environ.get("EVOAI_RESEARCH_ENABLE_WEB", "1").lower() in ("1", "true", "yes")
        self.fetch_timeout = float(os.environ.get("EVOAI_RESEARCH_TIMEOUT_SEC", "3.0"))
        raw_allowlist = os.environ.get("EVOAI_RESEARCH_ALLOWLIST", "")
        self.allowlist_hosts = [item.strip().lower() for item in raw_allowlist.split(",") if item.strip()]

    def _host_allowed(self, url: str) -> bool:
        if not self.allowlist_hosts:
            return True
        try:
            host = (urllib.parse.urlparse(url).hostname or "").lower()
        except Exception:
            return False
        return any(host == rule or host.endswith(f".{rule}") for rule in self.allowlist_hosts)

    def _fetch_url_excerpt(self, url: str) -> Dict[str, Any]:
        if not self._host_allowed(url):
            return {"url": url, "ok": False, "reason": "allowlist_blocked"}
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "EvoAI-ResearchToolkit/1.0",
                },
            )
            with urllib.request.urlopen(req, timeout=self.fetch_timeout) as resp:
                raw = resp.read(80_000)
            text = raw.decode("utf-8", errors="ignore")
            cleaned = re.sub(r"<[^>]+>", " ", text)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            excerpt = cleaned[:320]
            return {"url": url, "ok": True, "excerpt": excerpt}
        except urllib.error.URLError:
            return {"url": url, "ok": False, "reason": "fetch_failed"}
        except Exception:
            return {"url": url, "ok": False, "reason": "fetch_error"}

    def _plugin_research(self, query: str, engine) -> List[str]:
        findings: List[str] = []
        for plugin in getattr(engine, "plugins", []):
            try:
                if plugin.can_handle(query):
                    out = plugin.handle(query, engine)
                    if out:
                        findings.append(str(out))
            except Exception:
                continue
        return findings

    def research(self, query: str, engine) -> Dict[str, Any]:
        plugin_findings = self._plugin_research(query, engine)
        urls = self.URL_RE.findall(str(query or ""))[:3]

        web_results: List[Dict[str, Any]] = []
        if self.allow_web:
            for url in urls:
                web_results.append(self._fetch_url_excerpt(url))

        summary_parts: List[str] = []
        if plugin_findings:
            summary_parts.append(f"plugin findings: {len(plugin_findings)}")
        if web_results:
            ok_count = sum(1 for item in web_results if item.get("ok"))
            summary_parts.append(f"web fetches: {ok_count}/{len(web_results)} successful")
        if not summary_parts:
            summary_parts.append("no research sources available for this query")

        preview = ""
        if plugin_findings:
            preview = plugin_findings[0][:220]
        else:
            first_ok = next((item for item in web_results if item.get("ok") and item.get("excerpt")), None)
            if first_ok:
                preview = str(first_ok.get("excerpt", ""))[:220]

        if preview:
            summary = f"Research: {'; '.join(summary_parts)}. Top result: {preview}"
        else:
            summary = f"Research: {'; '.join(summary_parts)}."

        return {
            "summary": summary,
            "plugin_findings": plugin_findings,
            "web_results": web_results,
            "allow_web": self.allow_web,
        }
