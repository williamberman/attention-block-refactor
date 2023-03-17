import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_file", required=True, type=str)

    args = parser.parse_args()

    return args


def main(args):
    with open(args.output_file, "r") as f:
        outputs = json.load(f)

    processed = {}

    for repo_name, deprecated_classes in outputs.items():
        for deprecated_class in deprecated_classes:
            klass_name = deprecated_class["klass_name"]

            if klass_name not in processed:
                processed[klass_name] = 0
            else:
                processed[klass_name] += 1

    for klass_name, count in processed.items():
        print(f"{klass_name}: {count}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
