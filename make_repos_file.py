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

    output = []

    for diffusers_model in diffusers_models:
        model_id = diffusers_model.modelId

        if "lora" in diffusers_model.tags or "Lora" in diffusers_model.tags:
            tagged_as_lora = True
        else:
            tagged_as_lora = False

        output.append({"hub_upload_id": model_id, "tagged_as_lora": tagged_as_lora})

    with open(args.hub_uploads_file, "w") as f:
        json.dump(output, f, indent=4)


if __name__ == "__main__":
    args = parse_args()
    main(args)
