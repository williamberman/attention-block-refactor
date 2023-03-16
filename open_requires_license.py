import argparse
import json
import webbrowser


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--requires_licenses_file", required=True, type=str)

    args = parser.parse_args()

    return args


def main(args):
    chrome_path = "open -a /Applications/Google\ Chrome.app %s"
    browser = webbrowser.get(chrome_path)

    with open(args.requires_licenses_file) as f:
        requires_licenses = json.load(f)

    for requires_license in requires_licenses:
        requires_license = f"https://huggingface.co/{requires_license}"
        browser.open(requires_license)


if __name__ == "__main__":
    args = parse_args()
    main(args)
