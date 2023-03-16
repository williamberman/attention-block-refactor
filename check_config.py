import argparse
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import HfHubHTTPError
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

    parser.add_argument("--requires_license_output", required=True, type=str)

    parser.add_argument(
        "--hub_uploads_load_from_file", required=False, type=str, default=None
    )

    args = parser.parse_args()

    return args


def main(args):
    api = HfApi()

    if args.hub_uploads_load_from_file is None:
        hub_upload_ids = diffusers_hub_uploads(api)
    else:
        hub_upload_ids = []

        with open(args.hub_uploads_load_from_file) as f:
            for line in f.readlines():
                line = line.strip()

                if len(line) != 0:
                    hub_upload_ids.append(line)

    print(f"number diffusers hub uploads: {len(hub_upload_ids)}")

    hub_uploads_with_deprecated_attention_blocks = {}
    requires_license = []

    for hub_upload_id in hub_upload_ids:
        print(f"checking hub upload: {hub_upload_id}")

        try:
            snapshot_path = snapshot_download(
                hub_upload_id, allow_patterns="*.json", token=True
            )
        except HfHubHTTPError as e:
            if e.response.status_code == 403:
                print(f"requires license: {hub_upload_id}")
                requires_license.append(hub_upload_id)
                continue

            raise e

        models_with_deprecated_attention_blocks = find_deprecated_attention(
            snapshot_path
        )

        if models_with_deprecated_attention_blocks is None:
            continue

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

    requires_license = json.dumps(requires_license, indent=4)

    with open(args.output, "w") as f:
        f.write(hub_uploads_with_deprecated_attention_blocks)

    with open(args.requires_license_output, "w") as f:
        f.write(requires_license)


def diffusers_hub_uploads(api):
    # pull all diffusers compatible models from hub
    diffusers_models = api.list_models(filter="diffusers")

    hub_upload_ids = []

    for diffusers_model in diffusers_models:
        hub_id = diffusers_model.modelId
        hub_upload_ids.append(hub_id)

    return hub_upload_ids


def find_deprecated_attention(snapshot_path):
    model_index_path = os.path.join(snapshot_path, "model_index.json")
    nested_model_index_path = os.path.join(
        snapshot_path, "model_files", "model_index.json"
    )
    model_path = os.path.join(snapshot_path, "config.json")

    if os.path.isfile(model_index_path):
        models_with_deprecated_attention_blocks = load_from_model_index(
            snapshot_path=snapshot_path, model_index_path=model_index_path
        )
    elif os.path.isfile(nested_model_index_path):
        models_with_deprecated_attention_blocks = load_from_model_index(
            snapshot_path=snapshot_path,
            model_index_path=nested_model_index_path,
            nested_path="model_files",
        )
    elif os.path.isfile(model_path):
        return load_from_root_level_model(
            snapshot_path=snapshot_path, model_path=model_path
        )
    else:
        print(
            f"No root level `model_index.json` or `config.json` found: {snapshot_path}"
        )
        return None

    return models_with_deprecated_attention_blocks


def load_from_model_index(*args, snapshot_path, model_index_path, nested_path=None):
    with open(model_index_path) as f:
        model_index: dict = json.load(f)

    models_with_deprecated_attention_blocks = []

    for model_key, value in model_index.items():
        if not isinstance(value, list):
            continue

        klass_name = value[1]

        is_model_deprecated_ = is_model_deprecated(
            klass_name=klass_name,
            snapshot_path=snapshot_path,
            model_key=model_key,
            nested_path=nested_path,
        )

        if is_model_deprecated_:
            models_with_deprecated_attention_blocks.append(
                {"klass_name": klass_name, "model_key": model_key}
            )

    return models_with_deprecated_attention_blocks


def load_from_root_level_model(*args, snapshot_path, model_path):
    models_with_deprecated_attention_blocks = []

    with open(model_path) as f:
        model = json.load(f)

    klass_name = model["_class_name"]

    is_model_deprecated_ = is_model_deprecated(
        klass_name=klass_name, snapshot_path=snapshot_path
    )

    if is_model_deprecated_:
        models_with_deprecated_attention_blocks.append({"klass_name": klass_name})

    return models_with_deprecated_attention_blocks


def is_model_deprecated(
    *args, klass_name, snapshot_path, model_key=None, nested_path=None
):
    if klass_name not in ATTENTION_BLOCK_CLASSES:
        return False

    if nested_path is None:
        root_path = snapshot_path
    else:
        root_path = os.path.join(snapshot_path, nested_path)

    if model_key is not None and os.path.isdir(os.path.join(root_path, model_key)):
        klass_config_path = os.path.join(root_path, model_key, "config.json")
    else:
        klass_config_path = os.path.join(root_path, "config.json")

    if not os.path.isfile(klass_config_path):
        print(f"malformed diffusers repository for model {klass_config_path}")
        return False

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
