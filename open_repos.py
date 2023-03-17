import argparse
import json
import webbrowser


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--repos_file", required=True, type=str)

    args = parser.parse_args()

    return args


def main(args):
    chrome_path = "open -a /Applications/Google\ Chrome.app %s"
    browser = webbrowser.get(chrome_path)

    with open(args.repos_file) as f:
        repos = json.load(f)

    for repo in repos:
        repo = f"https://huggingface.co/{repo}"
        browser.open(repo)


if __name__ == "__main__":
    args = parse_args()
    main(args)
