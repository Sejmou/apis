from huggingface_hub import HfApi

api = HfApi()

models = api.list_models()
