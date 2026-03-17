import json
import sys
import urllib.error
import urllib.request


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: healthcheck.py <url>", file=sys.stderr)
        return 1

    request = urllib.request.Request(sys.argv[1], headers={"Accept": "application/json"})

    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            if response.status != 200:
                return 1
            payload = json.load(response)
    except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError):
        return 1

    if payload.get("status") != "healthy":
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
