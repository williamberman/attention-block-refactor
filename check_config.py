import argparse
from huggingface_hub import HfApi, snapshot_download, list_repo_files
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

UNET_1D_DOWN_BLOCKS = [
    "DownResnetBlock1D",
    "DownBlock1D",
    "AttnDownBlock1D",
    "DownBlock1DNoSkip",
]

UNET_1D_UP_BLOCKS = ["UpResnetBlock1D", "UpBlock1D", "AttnUpBlock1D", "UpBlock1DNoSkip"]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output", required=False, default=None, type=str)

    parser.add_argument(
        "--requires_license_output", required=False, default=None, type=str
    )

    parser.add_argument(
        "--malformed_repos_output", required=False, default=None, type=str
    )

    parser.add_argument(
        "--hub_uploads_load_from_file",
        required=False,
        type=str,
        default=None,
    )

    parser.add_argument(
        "--skip_file",
        required=False,
        type=str,
        default=None,
    )

    parser.add_argument("--check_repo", required=False, type=str, default=None)

    args = parser.parse_args()

    return args


def main(args):
    api = HfApi()

    skip_hub_ids = []
    if args.skip_file is not None:
        with open(args.skip_file) as f:
            skip_hub_ids = json.load(f)

    skip_hub_ids = set(skip_hub_ids)

    if args.check_repo is not None:
        hub_uploads = [{"hub_upload_id": args.check_repo, "tagged_as_lora": False}]
    elif args.hub_uploads_load_from_file is None:
        hub_uploads = diffusers_hub_uploads(api=api, skip_hub_ids=skip_hub_ids)
    else:
        hub_uploads = hub_uploads_from_file(
            filename=args.hub_uploads_load_from_file, skip_hub_ids=skip_hub_ids
        )

    print(f"number diffusers hub uploads: {len(hub_uploads)}")

    hub_uploads_with_deprecated_attention_blocks = {}
    requires_license = []
    malformed_repos = []

    for hub_upload in hub_upload:
        hub_upload_id = hub_upload["hub_upload_id"]

        is_tagged_lora_repository = hub_upload["tagged_as_lora"]

        print(f"checking hub upload: {hub_upload_id}")

        try:
            repo_files = list_repo_files(hub_upload_id, token=True)
        except HfHubHTTPError as e:
            if e.response.status_code == 403:
                print(f"requires license: {hub_upload_id}")
                requires_license.append(hub_upload_id)
                continue

            raise e

        is_full_pipeline_repository, nested_path = check_is_full_pipeline_repository(
            repo_files
        )

        is_root_level_model_repository = "config.json" in repo_files

        is_root_level_scheduler_repository = "scheduler_config.json" in repo_files

        is_lora_repository = "pytorch_lora_weights.bin" in repo_files

        is_custom_pipeline_repository = "pipeline.py" in repo_files

        is_empty_repo = check_is_empty_repo(repo_files)

        if is_empty_repo:
            print(f"empty repository {hub_upload_id}")
            continue

        if is_lora_repository:
            print(f"lora repository {hub_upload_id}")
            continue

        if (
            not is_full_pipeline_repository
            and not is_root_level_model_repository
            and is_custom_pipeline_repository
        ):
            print(f"custom pipeline repository: {hub_upload_id}")
            continue

        if (
            not is_full_pipeline_repository
            and not is_root_level_model_repository
            and is_root_level_scheduler_repository
        ):
            print(f"custom root level scheduler repository: {hub_upload_id}")
            continue

        if (
            not is_full_pipeline_repository
            and not is_root_level_model_repository
            and is_tagged_lora_repository
        ):
            print(f"tagged lora repository: {hub_upload_id}")
            continue

        if not is_full_pipeline_repository and not is_root_level_model_repository:
            print(f"malformed repo: {hub_upload_id}")
            malformed_repos.append(hub_upload_id)
            continue

        snapshot_path = snapshot_download(
            hub_upload_id, allow_patterns="*.json", token=True
        )

        if is_full_pipeline_repository:
            if nested_path is None:
                model_index_path = os.path.join(snapshot_path, "model_index.json")
            else:
                model_index_path = os.path.join(
                    snapshot_path, nested_path, "model_index.json"
                )

            models_with_deprecated_attention_blocks = load_from_model_index(
                snapshot_path=snapshot_path,
                model_index_path=model_index_path,
                nested_path=nested_path,
            )
        elif is_root_level_model_repository:
            model_path = os.path.join(snapshot_path, "config.json")

            models_with_deprecated_attention_blocks = load_from_root_level_model(
                snapshot_path=snapshot_path, model_path=model_path
            )
        else:
            assert False

        if len(models_with_deprecated_attention_blocks) == 0:
            print(f"no deprecated attention blocks in {hub_upload_id}")
            continue

        print(f"deprecated attention blocks in hub upload {hub_upload_id}")

        hub_uploads_with_deprecated_attention_blocks[
            hub_upload_id
        ] = models_with_deprecated_attention_blocks

    if args.output is None:
        print("skipping writing outputs")
    else:
        print(f"writing outputs to {args.output}")

        hub_uploads_with_deprecated_attention_blocks = json.dumps(
            hub_uploads_with_deprecated_attention_blocks, indent=4
        )

        with open(args.output, "w") as f:
            f.write(hub_uploads_with_deprecated_attention_blocks)

    if args.requires_license_output is None:
        print("skipping writing requires license")
    else:
        print(f"writing requires license to {args.requires_license_output}")

        requires_license = json.dumps(requires_license, indent=4)

        with open(args.requires_license_output, "w") as f:
            f.write(requires_license)

    if args.malformed_repos_output is None:
        print("skipping writing malformed repos")
    else:
        print(f"writing malformed repos to {args.malformed_repos_output}")

        malformed_repos = json.dumps(malformed_repos, indent=4)

        with open(args.malformed_repos_output, "w") as f:
            f.write(malformed_repos)


def diffusers_hub_uploads(*args, api, skip_hub_ids):
    # pull all diffusers compatible models from hub
    diffusers_models = api.list_models(filter="diffusers")

    hub_uploads = []

    for diffusers_model in diffusers_models:
        hub_id = diffusers_model.modelId

        if hub_id in skip_hub_ids:
            print(f"skipping {hub_id}")
            continue

        if "lora" in diffusers_model.tags or "Lora" in diffusers_model.tags:
            tagged_as_lora = True
        else:
            tagged_as_lora = False

        hub_uploads.append({"hub_upload_id": hub_id, "tagged_as_lora": tagged_as_lora})

    return hub_uploads


def hub_uploads_from_file(*args, filename, skip_hub_ids):
    hub_uploads = []

    with open(filename) as f:
        hub_uploads_ = json.load(f)

        for hub_upload in hub_uploads_:
            hub_upload_id = hub_upload["hub_upload_id"]

            if hub_upload_id in skip_hub_ids:
                print(f"skipping {hub_upload_id}")
                continue

            hub_uploads.append(hub_upload)

    return hub_uploads


def check_is_full_pipeline_repository(repo_files):
    is_full_pipeline_repository = False
    nested_path = None

    for repo_file in repo_files:
        path, filename = os.path.split(repo_file)

        if filename == "model_index.json":
            if path != "":
                nested_path = path
            is_full_pipeline_repository = True

    return is_full_pipeline_repository, nested_path


def check_is_empty_repo(repo_files):
    for file in repo_files:
        file_extension = os.path.splitext(file)[1]

        if (
            file != ".gitattributes"
            and file_extension != ".ckpt"
            and file_extension != ".png"
            and file_extension != ".jpg"
            and file_extension != ".jpeg"
            and file_extension != ".safetensors"
            and file_extension != ".safetensore"
            and file_extension != ".zip"
            and file_extension != ".pt"
            and file_extension != ".bin"
            and file_extension != ".txt"
            and file_extension != ".yaml"
            and file_extension != ".yml"
            and file_extension != ".pkl"
            and file_extension != ".sh"
            and file_extension != ".exe"
            and file_extension != ".md"
            and file_extension != ".wav"
            and file_extension != ".pkl"
        ):
            return False

    return True


def load_from_model_index(*args, snapshot_path, model_index_path, nested_path=None):
    with open(model_index_path) as f:
        try:
            model_index: dict = json.load(f)
        except json.decoder.JSONDecodeError as e:
            print(f"encountered bad json: {e.msg}")
            return []

    models_with_deprecated_attention_blocks = []

    for model_key, value in model_index.items():
        if not isinstance(value, list):
            continue

        if len(value) != 2:
            continue

        if value == [None, None]:
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
    with open(model_path) as f:
        try:
            model = json.load(f)
        except json.decoder.JSONDecodeError as e:
            print(f"encountered bad json: {e.msg}")
            return []

    models_with_deprecated_attention_blocks = []

    klass_name = model.get("_class_name", None)

    is_model_deprecated_ = is_model_deprecated(
        klass_name=klass_name, snapshot_path=snapshot_path
    )

    if is_model_deprecated_:
        models_with_deprecated_attention_blocks.append({"klass_name": klass_name})

    return models_with_deprecated_attention_blocks


def is_model_deprecated(
    *args, snapshot_path, model_key=None, nested_path=None, klass_name=None
):
    if klass_name is not None and klass_name not in ATTENTION_BLOCK_CLASSES:
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

    # HACK - the config file loaded for this model is for a CLIPTextEncoder regardless of
    # what model we think we should be looking at. Therefore, we can just early return.
    if "architectures" in klass_config and klass_config["architectures"] == [
        "CLIPTextModel"
    ]:
        return False

    down_block_types = klass_config.get("down_block_types", None)
    up_block_types = klass_config.get("up_block_types", None)

    if klass_name == "VQModel":
        # is always deprecated because of the encoder and decoder mid block
        return True
    elif klass_name == "AutoencoderKL":
        # is always deprecated because of the encoder and decoder mid block
        return True
    elif klass_name == "UNet2DModel":
        if down_block_types is None:
            # The default block types have deprecated attention blocks
            return True

        for down_block_type in down_block_types:
            if down_block_type in BLOCKS_WITH_DEPRECATED_ATTENTION:
                return True

        if up_block_types is None:
            # The default block types have deprecated attention blocks
            return True

        for up_block_type in up_block_types:
            if up_block_type in BLOCKS_WITH_DEPRECATED_ATTENTION:
                return True

        return False
    elif klass_name == "UNet2DConditionModel":
        if down_block_types is not None:
            for down_block_type in down_block_types:
                if down_block_type in BLOCKS_WITH_DEPRECATED_ATTENTION:
                    return True

        if up_block_types is not None:
            for up_block_type in up_block_types:
                if up_block_type in BLOCKS_WITH_DEPRECATED_ATTENTION:
                    return True

        return False
    elif klass_name is None:
        if is_model_1d_unet(
            down_block_types=down_block_types, up_block_types=up_block_types
        ):
            return False

        assert False
    else:
        assert False


def is_model_1d_unet(*args, down_block_types, up_block_types):
    if down_block_types is not None:
        for block_type in UNET_1D_DOWN_BLOCKS:
            if block_type in down_block_types:
                return True

    if up_block_types is not None:
        for block_type in UNET_1D_UP_BLOCKS:
            if block_type in up_block_types:
                return True

    return False


if __name__ == "__main__":
    args = parse_args()
    main(args)
