"""
extract_pr_features.py

Extracts static PR features from GitHub API, replicating the logic of
ckxkexing/pr-acceptance features_factory.

Feature groups (matching pr_datasets.csv columns):
  - Commit stats: add/delete/total lines (sum/max/min), file changes
  - PR info:      desc length, quality vs project mean/median, test/doc files
  - Contributor:  followers, commits, PRs, issues, projects (before PR date)
  - User-in-repo: first PR, merged/closed history, issue participation
  - Project:      PRs, commits, issues, comments (cumulative + last 31 days)

NOTE on user features: the original pipeline used GHTorrent (historical MySQL
archive). For old PRs those values reflect the state at submission time. This
script uses GitHub Search API, which for LIVE/RECENT PRs is equivalent, but
will differ for historical PRs where the user has since grown their activity.
"""

import datetime
import re
import sys
import time
from typing import Any, Optional

import requests


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_dt(s: str) -> datetime.datetime:
    return datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")


# ─────────────────────────────────────────────────────────────────────────────
# Extractor
# ─────────────────────────────────────────────────────────────────────────────

class PRFeatureExtractor:
    BASE = "https://api.github.com"

    def __init__(self, token: str):
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        })
        self.session.verify = False  # corporate SSL intercept

    # ── Low-level HTTP ───────────────────────────────────────────────────────

    def _get(self, path: str, params: dict = None) -> Any:
        url = self.BASE + path if path.startswith("/") else path
        for _ in range(4):
            r = self.session.get(url, params=params)
            if r.status_code == 403:
                reset = int(r.headers.get("X-RateLimit-Reset", time.time() + 60))
                wait = max(1, reset - time.time()) + 2
                print(f"  [rate-limit] waiting {wait:.0f}s ...", file=sys.stderr)
                time.sleep(wait)
                continue
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        raise RuntimeError("Rate-limit retries exhausted")

    def _paginate(self, path: str, params: dict = None) -> list:
        params = {**(params or {}), "per_page": 100, "page": 1}
        results = []
        while True:
            page = self._get(path, params)
            if not page:
                break
            results.extend(page)
            if len(page) < 100:
                break
            params = {**params, "page": params["page"] + 1}
        return results

    def _search_count(self, q: str, resource: str = "issues") -> int:
        """Return total_count from GitHub Search API. 1 req/s due to secondary limits."""
        r = self.session.get(
            f"{self.BASE}/search/{resource}",
            params={"q": q, "per_page": 1},
        )
        time.sleep(1.1)  # secondary rate limit: max 1 search/s
        if r.status_code in (422, 451):
            return 0
        if r.status_code == 403:
            reset = int(r.headers.get("X-RateLimit-Reset", time.time() + 60))
            time.sleep(max(1, reset - time.time()) + 2)
            return self._search_count(q, resource)
        r.raise_for_status()
        return r.json().get("total_count", 0)

    # ── Main extraction ──────────────────────────────────────────────────────

    def extract(self, owner: str, repo: str, pr_number: int) -> dict:
        repo_full = f"{owner}/{repo}"
        print(f"\nExtracting features for {repo_full} PR #{pr_number} ...")

        # ── 1. PR metadata ───────────────────────────────────────────────────
        print("  [1/8] PR metadata")
        pr = self._get(f"/repos/{repo_full}/pulls/{pr_number}")
        if pr is None:
            raise ValueError(f"PR #{pr_number} not found in {repo_full}")

        created_at  = _parse_dt(pr["created_at"])
        login       = pr["user"]["login"]
        pr_body     = pr.get("body") or ""
        pr_title    = pr.get("title") or ""

        # Date strings for filtering
        date_search     = created_at.strftime("%Y-%m-%d")
        month_ago       = (created_at - datetime.timedelta(days=31)).strftime("%Y-%m-%d")
        month_ago_dt    = created_at - datetime.timedelta(days=31)

        # ── 2. PR commits list ───────────────────────────────────────────────
        print("  [2/8] PR commits list")
        pr_commits = self._paginate(f"/repos/{repo_full}/pulls/{pr_number}/commits")
        pr_commit_count = len(pr_commits)

        # whether_pr_created_before_commit: PR opened before its first commit was pushed
        if pr_commits:
            first_commit_dt = _parse_dt(pr_commits[0]["commit"]["author"]["date"])
            whether_pr_created_before_commit = bool(created_at < first_commit_dt)
        else:
            whether_pr_created_before_commit = False

        # ── 3. Per-commit stats (1 API call per commit) ──────────────────────
        print(f"  [3/8] Per-commit stats ({pr_commit_count} commits)")
        adds, dels, files_per_commit = [], [], []
        contain_test_file = 0
        contain_doc_file  = 0

        for c in pr_commits:
            detail = self._get(f"/repos/{repo_full}/commits/{c['sha']}")
            if detail is None:
                continue
            stats = detail.get("stats", {})
            a = stats.get("additions", 0)
            d = stats.get("deletions", 0)
            adds.append(a)
            dels.append(d)
            commit_files = detail.get("files", [])
            files_per_commit.append(len(commit_files))
            for f in commit_files:
                name = f.get("filename", "").lower()
                if "test" in name:
                    contain_test_file += 1
                if "doc" in name:
                    contain_doc_file += 1

        def _agg(lst):
            if not lst:
                return 0, 0, 0
            return sum(lst), max(lst), min(lst)

        add_sum, add_max, add_min = _agg(adds)
        del_sum, del_max, del_min = _agg(dels)

        # total per commit: add_i + del_i  (NOT max(add)+max(del))
        totals         = [a + d for a, d in zip(adds, dels)]
        tot_sum, tot_max, tot_min = _agg(totals)

        # sum of files across all commits (original: counts each commit's files)
        commit_file_change = sum(files_per_commit)

        # ── 4. PR description features ───────────────────────────────────────
        print("  [4/8] PR description")
        pr_desc_len = len((pr_title + " " + pr_body).split())

        # check_pr_desc_mean / medium: is this PR's desc len >= running mean/median
        # over all PRs in the same repo that were created before this one.
        all_repo_prs = self._paginate(
            f"/repos/{repo_full}/pulls",
            {"state": "all", "sort": "created", "direction": "asc"}
        )
        prev_lens = []
        for p in all_repo_prs:
            if _parse_dt(p["created_at"]) >= created_at:
                break
            prev_lens.append(len(((p.get("title") or "") + " " + (p.get("body") or "")).split()))

        if prev_lens:
            mean_len   = sum(prev_lens) / len(prev_lens)
            s          = sorted(prev_lens)
            n          = len(s)
            median_len = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0
            check_pr_desc_mean   = bool(pr_desc_len >= mean_len)
            check_pr_desc_medium = bool(pr_desc_len >= median_len)
        else:
            check_pr_desc_mean   = False
            check_pr_desc_medium = False

        # ── 5. Bot detection ─────────────────────────────────────────────────
        # Logic from use_other/user_type.py: ends with [bot], -bot, -robot
        bot_user = 1 if re.search(r'[\W_]bot[\W_]?$|[\W_]robot$', login.lower()) else 0

        # ── 6. User features ─────────────────────────────────────────────────
        print("  [6/8] User features (GitHub Search)")
        user_data = self._get(f"/users/{login}")
        # NOTE: followers / projects are CURRENT values, not at-time-of-PR.
        # For live PRs this is essentially the same; for historical PRs it differs.
        before_pr_user_followers = float(user_data.get("followers", 0) if user_data else 0)
        before_pr_user_projects  = float(user_data.get("public_repos", 0) if user_data else 0)

        before_pr_user_commits      = self._search_count(
            f"author:{login} committer-date:<{date_search}", "commits"
        )
        before_pr_user_pulls        = self._search_count(
            f"type:pr author:{login} created:<{date_search}"
        )
        before_pr_user_issues       = self._search_count(
            f"type:issue author:{login} created:<{date_search}"
        )
        before_pr_user_commits_proj = self._search_count(
            f"repo:{repo_full} author:{login} committer-date:<{date_search}", "commits"
        )

        # ── 7. User-in-repo history ──────────────────────────────────────────
        print("  [7/8] User-in-repo history")
        n_created_before = self._search_count(
            f"repo:{repo_full} type:pr author:{login} created:<{date_search}"
        )
        number_of_created_pr_in_this_proj_before_pr = n_created_before
        is_this_proj_first = 1 if n_created_before == 0 else 0

        # Merged / closed by this user in this repo before PR date.
        # Original counted closed_at <= pr_created_at from all closed PRs.
        n_merged_by_user = 0
        n_closed_by_user = 0
        closed_prs = self._paginate(
            f"/repos/{repo_full}/pulls",
            {"state": "closed", "sort": "created", "direction": "asc"}
        )
        for p in closed_prs:
            if p["user"]["login"] != login:
                continue
            closed_str = p.get("closed_at") or p.get("updated_at")
            if closed_str and _parse_dt(closed_str) <= created_at:
                n_closed_by_user += 1
                if p.get("merged_at"):
                    n_merged_by_user += 1

        number_of_merged_pr_in_this_proj_before_pr = n_merged_by_user
        number_of_closed_pr_in_this_proj_before_pr = n_closed_by_user
        ratio_of_merged_pr_in_this_proj_before_pr  = (
            n_merged_by_user / n_closed_by_user if n_closed_by_user > 0 else 0.0
        )

        # Issue participation (created + joined via comment)
        issue_created_in_project_by_pr_author = self._search_count(
            f"repo:{repo_full} type:issue author:{login} created:<{date_search}"
        )
        issue_created_in_project_by_pr_author_in_month = self._search_count(
            f"repo:{repo_full} type:issue author:{login} created:{month_ago}..{date_search}"
        )
        # "joined" = commented on (including issues they didn't create)
        issue_joined_in_project_by_pr_author = self._search_count(
            f"repo:{repo_full} type:issue commenter:{login} created:<{date_search}"
        )
        issue_joined_in_project_by_pr_author_in_month = self._search_count(
            f"repo:{repo_full} type:issue commenter:{login} created:{month_ago}..{date_search}"
        )

        # ── 8. Project-level features ────────────────────────────────────────
        print("  [8/8] Project-level features")

        before_pr_project_prs = self._search_count(
            f"repo:{repo_full} type:pr created:<{date_search}"
        )
        before_pr_project_prs_in_month = self._search_count(
            f"repo:{repo_full} type:pr created:{month_ago}..{date_search}"
        )
        before_pr_merge_cnt = self._search_count(
            f"repo:{repo_full} type:pr is:merged closed:<{date_search}"
        )
        before_pr_closed_cnt = self._search_count(
            f"repo:{repo_full} type:pr is:unmerged closed:<{date_search}"
        )
        before_pr_merge_ratio = (
            before_pr_merge_cnt / (before_pr_merge_cnt + before_pr_closed_cnt)
            if (before_pr_merge_cnt + before_pr_closed_cnt) > 0 else 0.0
        )

        # Commits: search API gives total_count (original used cloned repo)
        before_pr_project_commits = self._search_count(
            f"repo:{repo_full} committer-date:<{date_search}", "commits"
        )
        before_pr_project_commits_in_month = self._search_count(
            f"repo:{repo_full} committer-date:{month_ago}..{date_search}", "commits"
        )

        before_pr_project_issues = self._search_count(
            f"repo:{repo_full} type:issue created:<{date_search}"
        )
        before_pr_project_issues_in_month = self._search_count(
            f"repo:{repo_full} type:issue created:{month_ago}..{date_search}"
        )

        # Issue comments + PR review comments (approx: sum `comments` field per issue/PR)
        # Original used MongoDB with per-comment timestamps. This sums the comments
        # field of items created before the PR date — a close approximation.
        before_pr_project_issues_comment          = 0
        before_pr_project_issues_comment_in_month = 0
        before_pr_project_comments_in_prs          = 0
        before_pr_project_comments_in_prs_in_month = 0

        all_issues = self._paginate(
            f"/repos/{repo_full}/issues",
            {"state": "all", "sort": "created", "direction": "asc"}
        )
        for iss in all_issues:
            iss_created = _parse_dt(iss["created_at"])
            if iss_created >= created_at:
                break  # sorted asc, no need to continue
            n_comments = iss.get("comments", 0)
            in_month   = iss_created >= month_ago_dt
            if "pull_request" in iss:
                before_pr_project_comments_in_prs += n_comments
                if in_month:
                    before_pr_project_comments_in_prs_in_month += n_comments
            else:
                before_pr_project_issues_comment += n_comments
                if in_month:
                    before_pr_project_issues_comment_in_month += n_comments

        # ── Assemble result ──────────────────────────────────────────────────
        return {
            "pr_url":   f"https://api.github.com/repos/{repo_full}/pulls/{pr_number}",
            "login":    login,
            # Commit stats
            "pr_commit_count":          pr_commit_count,
            "commit_add_line_sum":      add_sum,
            "commit_delete_line_sum":   del_sum,
            "commit_total_line_sum":    tot_sum,
            "commit_file_change":       commit_file_change,
            "commit_add_line_max":      add_max,
            "commit_delete_line_max":   del_max,
            "commit_total_line_max":    tot_max,
            "commit_add_line_min":      add_min,
            "commit_delete_line_min":   del_min,
            "commit_total_line_min":    tot_min,
            # PR description
            "pr_desc_len":          pr_desc_len,
            "contain_test_file":    float(contain_test_file),
            "contain_doc_file":     float(contain_doc_file),
            "check_pr_desc_mean":   check_pr_desc_mean,
            "check_pr_desc_medium": check_pr_desc_medium,
            # PR flags
            "is_this_proj_first":               is_this_proj_first,
            "whether_pr_created_before_commit": whether_pr_created_before_commit,
            "bot_user":                         bot_user,
            # User (GitHub profile, approximation for historical PRs)
            "before_pr_user_followers":    before_pr_user_followers,
            "before_pr_user_projects":     before_pr_user_projects,
            "before_pr_user_commits":      before_pr_user_commits,
            "before_pr_user_pulls":        before_pr_user_pulls,
            "before_pr_user_issues":       before_pr_user_issues,
            "before_pr_user_commits_proj": before_pr_user_commits_proj,
            # User-in-repo
            "number_of_created_pr_in_this_proj_before_pr":  number_of_created_pr_in_this_proj_before_pr,
            "number_of_merged_pr_in_this_proj_before_pr":   number_of_merged_pr_in_this_proj_before_pr,
            "number_of_closed_pr_in_this_proj_before_pr":   number_of_closed_pr_in_this_proj_before_pr,
            "ratio_of_merged_pr_in_this_proj_before_pr":    ratio_of_merged_pr_in_this_proj_before_pr,
            "issue_created_in_project_by_pr_author":             issue_created_in_project_by_pr_author,
            "issue_created_in_project_by_pr_author_in_month":   issue_created_in_project_by_pr_author_in_month,
            "issue_joined_in_project_by_pr_author":              issue_joined_in_project_by_pr_author,
            "issue_joined_in_project_by_pr_author_in_month":    issue_joined_in_project_by_pr_author_in_month,
            # Project
            "before_pr_project_prs":                     before_pr_project_prs,
            "before_pr_project_prs_in_month":            before_pr_project_prs_in_month,
            "before_pr_merge_cnt":                        before_pr_merge_cnt,
            "before_pr_closed_cnt":                       before_pr_closed_cnt,
            "before_pr_merge_ratio":                      before_pr_merge_ratio,
            "before_pr_project_commits":                  before_pr_project_commits,
            "before_pr_project_commits_in_month":         before_pr_project_commits_in_month,
            "before_pr_project_issues":                   before_pr_project_issues,
            "before_pr_project_issues_in_month":          before_pr_project_issues_in_month,
            "before_pr_project_issues_comment":           before_pr_project_issues_comment,
            "before_pr_project_issues_comment_in_month":  before_pr_project_issues_comment_in_month,
            "before_pr_project_comments_in_prs":          before_pr_project_comments_in_prs,
            "before_pr_project_comments_in_prs_in_month": before_pr_project_comments_in_prs_in_month,
        }


# ─────────────────────────────────────────────────────────────────────────────
# CLI: test against flask PR #113 (first row of pr_datasets.csv)
# ─────────────────────────────────────────────────────────────────────────────

DATASET_REFERENCE = {
    "pr_commit_count":          30.0,
    "commit_add_line_sum":      1855.0,
    "commit_delete_line_sum":   7205.0,
    "commit_total_line_sum":    9060.0,
    "commit_file_change":       154.0,
    "commit_add_line_max":      825.0,
    "commit_delete_line_max":   6930.0,
    "commit_total_line_max":    6931.0,
    "commit_add_line_min":      0.0,
    "commit_delete_line_min":   0.0,
    "commit_total_line_min":    0.0,
    "pr_desc_len":              30,
    "contain_test_file":        5.0,
    "contain_doc_file":         47.0,
    "check_pr_desc_mean":       False,
    "check_pr_desc_medium":     False,
    "is_this_proj_first":       1,
    "whether_pr_created_before_commit": False,
    "bot_user":                 0,
    # User features — GHTorrent historical values (will differ for old PRs)
    "before_pr_user_followers":     1.0,
    "before_pr_user_projects":      1.0,
    "before_pr_user_commits":       4,
    "before_pr_user_pulls":         1,
    "before_pr_user_issues":        5,
    "before_pr_user_commits_proj":  3,
    # User-in-repo
    "number_of_created_pr_in_this_proj_before_pr":  0,
    "number_of_merged_pr_in_this_proj_before_pr":   0,
    "number_of_closed_pr_in_this_proj_before_pr":   0,
    "ratio_of_merged_pr_in_this_proj_before_pr":    0.0,
    "issue_created_in_project_by_pr_author":            4,
    "issue_created_in_project_by_pr_author_in_month":   0,
    "issue_joined_in_project_by_pr_author":             6,
    "issue_joined_in_project_by_pr_author_in_month":    0,
    # Project
    "before_pr_project_prs":            1,
    "before_pr_project_prs_in_month":   1,
    "before_pr_merge_cnt":              0,
    "before_pr_closed_cnt":             0,
    "before_pr_merge_ratio":            0.0,
    "before_pr_project_commits":        535,
    "before_pr_project_commits_in_month": 26,
    "before_pr_project_issues":         111,
    "before_pr_project_issues_in_month": 11,
    "before_pr_project_issues_comment":           242,
    "before_pr_project_issues_comment_in_month":  30,
    "before_pr_project_comments_in_prs":          1,
    "before_pr_project_comments_in_prs_in_month": 1,
}

HISTORICAL_FEATURES = {
    "before_pr_user_followers", "before_pr_user_projects",
    "before_pr_user_commits", "before_pr_user_pulls",
    "before_pr_user_issues", "before_pr_user_commits_proj",
}


def _matches(got, expected) -> bool:
    try:
        return abs(float(got) - float(expected)) < 0.5
    except (TypeError, ValueError):
        return str(got) == str(expected)


if __name__ == "__main__":
    import os

    token = os.environ.get("GITHUB_TOKEN") or (
        open(os.path.expanduser("~/.github_token")).read().strip()
        if os.path.exists(os.path.expanduser("~/.github_token")) else None
    )
    if not token:
        print("ERROR: set GITHUB_TOKEN env var or put token in ~/.github_token",
              file=sys.stderr)
        sys.exit(1)

    ex = PRFeatureExtractor(token)
    features = ex.extract("pallets", "flask", 113)

    # ── Print comparison table ───────────────────────────────────────────────
    print("\n" + "=" * 90)
    print(f"  {'FEATURE':<52} {'DATASET':>10} {'EXTRACTED':>10}  {'OK?'}")
    print("=" * 90)

    n_ok = n_diff = n_hist = 0
    for feat, expected in DATASET_REFERENCE.items():
        got    = features.get(feat, "N/A")
        ok     = _matches(got, expected)
        is_his = feat in HISTORICAL_FEATURES
        if is_his:
            status = "(historical~)"
            n_hist += 1
        elif ok:
            status = "OK"
            n_ok += 1
        else:
            status = "DIFF"
            n_diff += 1
        print(f"  {feat:<52} {str(expected):>10} {str(got):>10}  {status}")

    print("=" * 90)
    print(f"  Exact match: {n_ok}  |  Diff: {n_diff}  |  Historical approx: {n_hist}")
    print()
