from __future__ import annotations

import requests

from common import load_config, make_lead, parse_args, write_leads


def main() -> None:
    args = parse_args("Collect Reddit posts from configured subreddits and queries.")
    config = load_config(args.config)
    headers = {"User-Agent": "FounderRadar/0.1 personal scout tool"}
    leads = []
    seen = set()

    for watch in config.get("reddit", []):
        subreddit = watch["subreddit"].removeprefix("r/")
        for query in watch.get("queries", []):
            url = f"https://www.reddit.com/r/{subreddit}/search.json"
            response = requests.get(
                url,
                params={"q": query, "restrict_sr": 1, "sort": "new", "limit": min(args.limit, 25)},
                headers=headers,
                timeout=30,
            )
            response.raise_for_status()
            for child in response.json().get("data", {}).get("children", []):
                post = child.get("data", {})
                post_id = post.get("id")
                if not post_id or post_id in seen:
                    continue
                seen.add(post_id)
                title = post.get("title", "")
                body = post.get("selftext", "")
                author = post.get("author", "Unknown Redditor")
                permalink = f"https://www.reddit.com{post.get('permalink', '')}"
                summary = f"{title}. {body[:280]}"
                leads.append(
                    make_lead(
                        name=author,
                        source="Reddit",
                        summary=summary,
                        link=permalink,
                        notes=f"Matched r/{subreddit} query: {query}. Score: {post.get('score', 0)}. Comments: {post.get('num_comments', 0)}.",
                        external_id=f"reddit:{post_id}",
                    )
                )

    write_leads(args.out, "Reddit", leads)


if __name__ == "__main__":
    main()
