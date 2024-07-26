from huggingface_hub.hf_api import ModelInfo, ExpandModelProperty_T
from tqdm import tqdm
from huggingface_hub import HfApi, file_exists, hf_hub_download
from huggingface_hub.utils._errors import GatedRepoError
from typing import List, Set
import json
import numpy as np
from datetime import datetime, date
import os
from dataclasses import asdict
from requests.exceptions import HTTPError
from typing import List

data_dir = "data"
os.makedirs(data_dir, exist_ok=True)
api = HfApi()

MODEL_ID_FILE = os.path.join(data_dir, "relevant_model_ids.txt")
MODELS_FILE = os.path.join(data_dir, "models.jsonl")
ACCESS_RESTRICTED_MODELS_FILE = os.path.join(data_dir, "access_restricted_models.txt")
MODEL_CONFIGS_FILE = os.path.join(data_dir, "model_configs.jsonl")
MODEL_IDS_WO_CONFIG_FILE = os.path.join(data_dir, "models_without_config.txt")
MODEL_IDS_WITH_CONFIG_FETCHING_ERRORS_FILE = os.path.join(data_dir, "models_with_config_fetching_errors.txt")

nlp_category_filters = [
    "text-classification",
    "token-classification",
    "table-question-answering",
    "question-answering",
    "zero-shot-classification",
    "translation",
    "summarization",
    "feature-extraction",
    "text-generation",
    "text2text-generation",
    "fill-mask",
    "sentence-similarity",
]

# need to specify all additional fields we want to fetch by passing them in the `expand` parameter
# see also: https://huggingface.co/docs/huggingface_hub/v0.24.2/en/package_reference/hf_api#huggingface_hub.hf_api.ModelInfo:~:text=to%20False.-,expand,-(List%5BExpandModelProperty_T
EXPAND_PARAMS: List[ExpandModelProperty_T] = [
    "author",
    "cardData",
    "config",
    "createdAt",
    "disabled",
    "downloads",
    "downloadsAllTime",
    "gated",
    "inference",
    "lastModified",
    "library_name",
    "likes",
    "mask_token",
    "model-index",
    "pipeline_tag",
    "private",
    "safetensors",
    "sha",
    "siblings",
    "spaces",
    "tags",
    "transformersInfo",
    "widgetData",
]

def get_missing_model_ids():
    if os.path.exists(MODEL_ID_FILE):
        print(f"Reading model IDs from {MODEL_ID_FILE}")
        with open(MODEL_ID_FILE, "r") as f:
            model_ids = [line.strip() for line in f.readlines() if line.strip()]
    else:
        model_ids = download_relevant_model_ids()
        write_to_file(model_ids, MODEL_ID_FILE)

    return model_ids

def download_relevant_model_ids() -> List[str]:
    models_found: List[ModelInfo] = []
    for cat_filter in tqdm(nlp_category_filters, desc="Fetching models with at least one like for each category"):
        returned_models = api.list_models(filter=cat_filter, sort="likes")
        models_matching_criteria: List[ModelInfo] = []
        for model in returned_models:
            if model.likes is not None and model.likes > 0:
                models_matching_criteria.append(model)
            else:
                break
        print(f"{len(models_matching_criteria)} models found for category {cat_filter}")
        models_found.extend(models_matching_criteria)

    relevant_model_ids = [model.id for model in models_found]

    return relevant_model_ids


def write_to_file(strings: List[str], filepath: str):
    with open(filepath, "w") as f:
        for string in strings:
            f.write(f"{string}\n")


def get_current_timestamp_str():
    return datetime.strftime(datetime.now(), "%Y-%m-%dT%H:%M:%S.000")


class SafeJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that also supports numpy data types (see https://stackoverflow.com/a/57915246/13727176) and date/datetime objects.
    """

    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, datetime):
            return o.isoformat()
        if isinstance(o, date):
            return o.isoformat()
        return super(SafeJSONEncoder, self).default(o)


def safe_convert_to_json(data: dict):
    """
    A safer way to convert a dictionary to a JSON string, using our custom SafeJSONEncoder
    which supports numpy data types as well as date and datetime, transforming each into JSON-serializable equivalents.
    """
    return json.dumps(data, cls=SafeJSONEncoder)


def download_model_data(
    model_ids: List[str], expand_params: List[ExpandModelProperty_T] = EXPAND_PARAMS
):
    print(f"Got {len(model_ids)} model IDs")
    
    already_loaded_ids = set()
    if os.path.exists(MODELS_FILE):
        print(f"Reading already loaded model IDs from {MODELS_FILE}")
        with open(MODELS_FILE, "r") as f:
            already_loaded_ids = set([json.loads(line)["id"] for line in f.readlines()])
            print(f"Already loaded data for {len(already_loaded_ids)} models")

    access_restricted_model_ids = set()
    if os.path.exists(ACCESS_RESTRICTED_MODELS_FILE):
        print(f"Reading access restricted model IDs from {ACCESS_RESTRICTED_MODELS_FILE}")
        with open(ACCESS_RESTRICTED_MODELS_FILE, "r") as f:
            access_restricted_model_ids = set([line.strip() for line in f.readlines()])
            print(f"Found {len(access_restricted_model_ids)} models that are access restricted")

    ids_to_download = set(model_ids) - already_loaded_ids - access_restricted_model_ids
    print(f"{len(ids_to_download)} models left to download")

    with open(MODELS_FILE, "a") as f:
        for model_id in tqdm(ids_to_download):
            try:
                model = api.model_info(model_id, expand=expand_params)
                model_dict = asdict(model)
                model_dict["observed_at"] = get_current_timestamp_str()
                f.write(safe_convert_to_json(model_dict) + "\n")
            except HTTPError as e:
                if e.response.status_code == 401:
                    with open(ACCESS_RESTRICTED_MODELS_FILE, "a") as err_f:
                        err_f.write(model_id + "\n")
                    continue
                else:
                    raise


def download_config_file(
    model_id: str, model_ids_with_configs: set, model_ids_without_config: set, access_restricted_model_ids: set
):
    try:
        config_exists = file_exists(model_id, "config.json")
    except GatedRepoError as e:
        with open(ACCESS_RESTRICTED_MODELS_FILE, "a") as err_f:
            err_f.write(model_id + "\n")
        access_restricted_model_ids.add(model_id)
        return
    if not config_exists:
        print(f"Config file for model {model_id} does not exist")
        with open(MODEL_IDS_WO_CONFIG_FILE, "a") as f:
            f.write(model_id + "\n")
        model_ids_without_config.add(model_id)
        return
    if model_id in model_ids_with_configs or model_id in access_restricted_model_ids:
        return
    tmp_dir = os.path.join(data_dir, "tmp")
    try: 
        hf_hub_download(model_id, "config.json", local_dir=tmp_dir, force_download=True)
        downloaded_path = os.path.abspath(os.path.join(tmp_dir, "config.json"))
        with open(downloaded_path, "r") as f:
            config = json.load(f)
            if "model_id" in config:
                raise ValueError("The config file already contains a 'model_id' field")
            config["model_id"] = model_id
        with open(MODEL_CONFIGS_FILE, "a") as f:
            f.write(safe_convert_to_json(config) + "\n")
            model_ids_with_configs.add(model_id)
    except GatedRepoError as e:
        with open(ACCESS_RESTRICTED_MODELS_FILE, "a") as err_f:
            err_f.write(model_id + "\n")
            access_restricted_model_ids.add(model_id)
    except HTTPError as e:
        if e.response.status_code == 401 or e.response.status_code == 403:
            with open(ACCESS_RESTRICTED_MODELS_FILE, "a") as err_f:
                err_f.write(model_id + "\n")
                access_restricted_model_ids.add(model_id)
        else:
            raise
    except Exception as e:
        print(f"Unknown error fetching config for model {model_id}: {e}")
        with open(MODEL_IDS_WITH_CONFIG_FETCHING_ERRORS_FILE, "a") as err_f:
            err_f.write(model_id + "\n")


def download_missing_model_configs(model_ids: Set[str]):
    model_ids_with_config = set()
    if os.path.exists(MODEL_CONFIGS_FILE):
        print(f"Reading model IDs with config from {MODEL_CONFIGS_FILE}")
        with open(MODEL_CONFIGS_FILE, "r") as f:
            model_ids_with_config = set(
                [json.loads(line)["model_id"] for line in f.readlines()]
            )
            print(f"Already loaded config data for {len(model_ids_with_config)} models")

    model_ids_without_config = set()
    if os.path.exists(MODEL_IDS_WO_CONFIG_FILE):
        print(f"Reading model IDs without config from {MODEL_IDS_WO_CONFIG_FILE}")
        with open(MODEL_IDS_WO_CONFIG_FILE, "r") as f:
            model_ids_without_config = set([line.strip() for line in f.readlines()])
            print(f"Found {len(model_ids_without_config)} models without config data")

    access_restricted_model_ids = set()
    if os.path.exists(ACCESS_RESTRICTED_MODELS_FILE):
        print(f"Reading access restricted model IDs from {ACCESS_RESTRICTED_MODELS_FILE}")
        with open(ACCESS_RESTRICTED_MODELS_FILE, "r") as f:
            access_restricted_model_ids = set([line.strip() for line in f.readlines()])
            print(f"Found {len(access_restricted_model_ids)} models that are access restricted")

    model_ids_to_fetch_config_for = (
        model_ids - model_ids_with_config - model_ids_without_config - access_restricted_model_ids
    )
    print(f"{len(model_ids_to_fetch_config_for)} models left to fetch config for")

    for model_id in tqdm(model_ids_to_fetch_config_for):
        download_config_file(
            model_id,
            model_ids_with_configs=model_ids_with_config,
            model_ids_without_config=model_ids_without_config,
            access_restricted_model_ids=access_restricted_model_ids,
        )

if __name__ == "__main__":
    model_ids = get_missing_model_ids()
    download_model_data(model_ids, EXPAND_PARAMS)
    download_missing_model_configs(set(model_ids))