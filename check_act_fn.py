from huggingface_hub import HfApi
import argparse
from huggingface_hub import HfApi, snapshot_download, list_repo_files
from huggingface_hub.utils import HfHubHTTPError
import json
import os

def main():
    diffusers_repos = get_diffusers_repos()

    requires_license = []
    malformed_repos = []
    alternative_act_fn = []

    for repo in diffusers_repos:
        hub_upload_id = repo["hub_upload_id"]
        is_tagged_lora_repository = repo["tagged_as_lora"]

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

            uses_silu = load_from_model_index(
                snapshot_path=snapshot_path,
                model_index_path=model_index_path,
                nested_path=nested_path,
            )

        elif is_root_level_model_repository:
            model_path = os.path.join(snapshot_path, "config.json")

            uses_silu = load_from_root_level_model(
                snapshot_path=snapshot_path, model_path=model_path
            )

        else:
            assert False

        if uses_silu is None:
            print(f"conditional unet activation function does not apply {hub_upload_id}")
        elif uses_silu == True:
            print(f"uses silu: {hub_upload_id}")
        elif uses_silu == False:
            print(f"uses alternative act_fn: {hub_upload_id}")
            alternative_act_fn.append(hub_upload_id)
            break
        else:
            assert False




def get_diffusers_repos():
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

    return output

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
            return None

    uses_silu = []

    for model_key, value in model_index.items():
        if not isinstance(value, list):
            continue

        if len(value) != 2:
            continue

        if value == [None, None]:
            continue

        klass_name = value[1]

        if klass_name != "UNet2DConditionModel":
            continue

        if nested_path is not None:
            root_path = os.path.join(snapshot_path, nested_path)
        else:
            root_path = snapshot_path

        if os.path.isdir(os.path.join(root_path, model_key)):
            klass_config_path = os.path.join(root_path, model_key, "config.json")

        uses_silu_ = does_unet_config_use_silu(klass_config_path)

        uses_silu.append(uses_silu_)

    if len(uses_silu) == 0:
        return None

    any_uses_silu = any([uses_silu_ == True for uses_silu_ in uses_silu])
    any_uses_non_silu = any([uses_silu_ == False for uses_silu_ in uses_silu])

    if any_uses_silu:
        return True
    elif any_uses_non_silu:
        return False
    else:
        return None


def load_from_root_level_model(*args, snapshot_path, model_path):
    with open(model_path) as f:
        try:
            model = json.load(f)
        except json.decoder.JSONDecodeError as e:
            print(f"encountered bad json: {e.msg}")
            return None

    klass_name = model.get("_class_name", None)

    if klass_name != "UNet2DConditionModel":
        return None

    root_path = snapshot_path
    klass_config_path = os.path.join(root_path, "config.json")

    uses_silu = does_unet_config_use_silu(klass_config_path)

    return uses_silu


def does_unet_config_use_silu(klass_config_path):
    with open(klass_config_path) as f:
        klass_config: dict = json.load(f)

    # HACK - the config file loaded for this model is for a CLIPTextEncoder regardless of
    # what model we think we should be looking at. Therefore, we can just early return.
    if "architectures" in klass_config and klass_config["architectures"] == [
        "CLIPTextModel"
    ]:
        return None

    act_fn = klass_config.get("act_fn", "silu")

    return act_fn == "silu"

if __name__ == "__main__":
    main()