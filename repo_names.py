from huggingface_hub import HfApi
import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--hub_uploads_file", required=True, type=str)

    args = parser.parse_args()

    return args


def main(args):
    api = HfApi()

    diffusers_models = api.list_models(filter="diffusers")

    model_ids = []

    for diffusers_model in diffusers_models:
        model_id = diffusers_model.modelId

        model_ids.append(model_id)

    with open(args.hub_uploads_file, "w") as f:
        json.dump(model_ids, f, indent=4)


if __name__ == "__main__":
    args = parse_args()
    main(args)
