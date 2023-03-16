from huggingface_hub import HfApi

api = HfApi()

# pull all diffusers compatible models from hub
diffusers_models = api.list_models(filter="diffusers")

for diffusers_model in diffusers_models:
    model_id = diffusers_model.modelId
    print(model_id)
