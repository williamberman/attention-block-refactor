import argparse
from huggingface_hub import HfApi, snapshot_download
import json
import os

ATTENTION_BLOCK_CLASSES = [
    "VQModel",
    "AutoencoderKL",
    "UNet2DModel",
    "UNet2DConditionModel",
]

BLOCKS_WITH_DEPRECATED_ATTENTION = [
    "AttnDownBlock2D",
    "AttnSkipDownBlock2D",
    "AttnDownEncoderBlock2D",
    "AttnUpBlock2D",
    "AttnSkipUpBlock2D",
    "AttnUpDecoderBlock2D",
    "UNetMidBlock2D",
]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output", required=True, type=str)

    args = parser.parse_args()

    return args


def main(args):
    api = HfApi()

    hub_upload_ids = diffusers_hub_uploads(api)

    print(f"number diffusers hub uploads: {len(hub_upload_ids)}")

    hub_uploads_with_deprecated_attention_blocks = {}

    for hub_upload_id in hub_upload_ids:
        print(f"checking hub upload: {hub_upload_id}")

        models_with_deprecated_attention_blocks = find_deprecated_attention(
            hub_upload_id
        )

        if len(models_with_deprecated_attention_blocks) == 0:
            print(f"no deprecated attention blocks in {hub_upload_id}")
            continue

        print(f"deprecated attention blocks in hub upload {hub_upload_id}")

        hub_uploads_with_deprecated_attention_blocks[
            hub_upload_id
        ] = models_with_deprecated_attention_blocks

    hub_uploads_with_deprecated_attention_blocks = json.dumps(
        hub_uploads_with_deprecated_attention_blocks, indent=4
    )

    with open(args.output, "w") as outfile:
        outfile.write(hub_uploads_with_deprecated_attention_blocks)


def diffusers_hub_uploads(api):
    # pull all diffusers compatible models from hub
    diffusers_models = api.list_models(filter="diffusers")

    hub_upload_ids = []

    for diffusers_model in diffusers_models:
        hub_id = diffusers_model.modelId
        hub_upload_ids.append(hub_id)

    return hub_upload_ids


def find_deprecated_attention(hub_upload_id):
    models_with_deprecated_attention_blocks = []

    snapshot_path = snapshot_download(hub_upload_id, allow_patterns="*.json")

    model_index_path = os.path.join(snapshot_path, "model_index.json")

    if os.path.isfile(model_index_path):
        with open(model_index_path) as f:
            model_index: dict = json.load(f)

        for model_key, value in model_index.items():
            if not isinstance(value, list):
                continue

            klass_name = value[1]

            is_model_deprecated_ = is_model_deprecated(
                klass_name=klass_name, snapshot_path=snapshot_path, model_key=model_key
            )

            if is_model_deprecated_:
                models_with_deprecated_attention_blocks.append(
                    {"klass_name": klass_name, "model_key": model_key}
                )
    else:
        model_path = os.path.join(snapshot_path, "config.json")

        if os.path.isfile(model_path):
            # root level model

            with open(model_path) as f:
                model = json.load(f)

            klass_name = model["_class_name"]

            is_model_deprecated_ = is_model_deprecated(
                klass_name=klass_name, snapshot_path=snapshot_path
            )

            if is_model_deprecated_:
                models_with_deprecated_attention_blocks.append(
                    {"klass_name": klass_name}
                )

        else:
            raise ValueError("No root level `model_index.json` or `config.json` found")

    return models_with_deprecated_attention_blocks


def is_model_deprecated(*args, klass_name, snapshot_path, model_key=None):
    if klass_name not in ATTENTION_BLOCK_CLASSES:
        return False

    if model_key is not None and os.path.isdir(os.path.join(snapshot_path, model_key)):
        klass_config_path = os.path.join(snapshot_path, model_key, "config.json")
    else:
        klass_config_path = os.path.join(snapshot_path, "config.json")

    with open(klass_config_path) as f:
        klass_config: dict = json.load(f)

    down_block_types = klass_config["down_block_types"]
    up_block_types = klass_config["up_block_types"]

    deprecated_attention_block = False

    for down_block_type in down_block_types:
        if down_block_type in BLOCKS_WITH_DEPRECATED_ATTENTION:
            deprecated_attention_block = True

    for up_block_type in up_block_types:
        if up_block_type in BLOCKS_WITH_DEPRECATED_ATTENTION:
            deprecated_attention_block = True

    if deprecated_attention_block:
        return True


if __name__ == "__main__":
    args = parse_args()
    main(args)
