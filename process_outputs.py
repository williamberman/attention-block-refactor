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
            deprecated_blocks = deprecated_class["deprecated_blocks"]

            if klass_name not in processed:
                processed[klass_name] = {
                    "count": 0,
                    "deprecated_blocks": {}
                }

            processed[klass_name]["count"] += 1

            for deprecated_block in deprecated_blocks:
                if deprecated_block not in processed[klass_name]["deprecated_blocks"]:
                    processed[klass_name]["deprecated_blocks"][deprecated_block] = 0

                processed[klass_name]["deprecated_blocks"][deprecated_block] += 1

    for klass_name, processed_ in processed.items():
        count = processed_["count"]

        out_str = f"{klass_name}: {count}"

        for deprecated_block, deprecated_block_count in processed_["deprecated_blocks"].items():
            out_str = f"{out_str}, {deprecated_block}: {deprecated_block_count}"

        print(out_str)


if __name__ == "__main__":
    args = parse_args()
    main(args)
