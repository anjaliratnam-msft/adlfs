import logging
import json
import os
import tempfile
import time
from io import BytesIO

import hydra
import hydra.core.hydra_config
import torch

from azstoragetorch.io import BlobIO
from adlfs import AzureBlobFileSystem

LOGGER = logging.getLogger(__name__)

ROOTDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_MODELS_DIR = os.environ.get(
    "AZSTORAGETORCH_LOCAL_MODELS", os.path.join(ROOTDIR, "local-models")
)


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def write_perf(cfg):
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    write_result_metadata(results_dir, cfg)
    content = preload_model(cfg)
    
    for i in range(cfg["num-runs"]):
        LOGGER.info("Run %s - Saving model for run", i)
        start = time.time()
        with get_writeable_file(cfg) as f:
            if cfg["write-method"]["name"] == "torch-save":
                torch.save(content, f)
            elif cfg["write-method"]["name"] == "writeall":
                print("writing")
                f.write(content)
            elif cfg["write-method"]["name"] == "write-partial":
                for j in range(0, len(content), 1024):
                    f.write(content[j:j + 1024])
        duration = time.time() - start
        del f
        LOGGER.info(f"Run %s - Seconds to write model: %s", i, duration)
        with open(os.path.join(results_dir, f"{i}.txt"), "w") as f:
            f.write(f"{duration}\n")
        print(duration)


def write_result_metadata(results_dir, cfg):
    with open(os.path.join(results_dir, "metadata.json"), "w") as f:
        f.write(json.dumps(
            {
                'write-method': cfg["write-method"]["name"],
                'filelike-impl': cfg["filelike-impl"]["name"],
                'model-size': cfg["model"]["size"],
                'num-runs': cfg["num-runs"],
                'block-size': cfg["block-size"]["size"],
            },
            indent=2,
        ))


def get_writeable_file(cfg, **kwargs):
    filelike_impl_name = cfg["filelike-impl"]["name"]
    if filelike_impl_name.startswith("blobio"):
        blob_url = get_blob_url(cfg["blob"], cfg["model"]["name"])
        sas_token = retrieve_sas_token(cfg)
        if sas_token is not None:
            blob_url += "?" + sas_token
        blob_io = BlobIO(blob_url, mode="wb")
        return blob_io
    elif filelike_impl_name == "open":
        return tempfile.TemporaryFile("wb")
    elif filelike_impl_name == "bytesio":
        return BytesIO()
    elif filelike_impl_name == "adlfs":
        fs = AzureBlobFileSystem(
            account_name=cfg["blob"]["account"],
            connection_string=os.getenv("AZURE_STORAGE_CONNECTION_STRING"),
            blocksize=cfg["block-size"]["size"],
        )
        return fs.open(f"az://{cfg['blob']['container']}/write-output/{cfg['model']['name']}", "wb")
    raise ValueError(f"Unknown filelike-impl: {cfg['filelike-impl']}")


def open_local_model(cfg):
    return open(os.path.join(LOCAL_MODELS_DIR, cfg["model"]["name"]), "rb")

def preload_model(cfg):
    if cfg["model"]["name"] == "small-phi-4.pth":
        blob_url = f"https://{os.getenv("AZURE_STORAGE_ACCOUNT_NAME")}.blob.core.windows.net/perf/{cfg['model']['name']}"
        sas_token = retrieve_sas_token(cfg)
        blob_url += "?" + sas_token
        with BlobIO(blob_url, mode="rb") as f:
            return torch.load(f, weights_only=True)
    elif cfg["model"]["name"] == "27GB.txt":
        blob_url = f"https://{os.getenv("AZURE_STORAGE_ACCOUNT_NAME")}.blob.core.windows.net/perf/{cfg['model']['name']}"
        sas_token = retrieve_sas_token(cfg)
        blob_url += "?" + sas_token
        with BlobIO(blob_url, mode="rb") as f:
            return f.read(18 * 1024**3) 
    elif cfg["write-method"]["name"] == "torch-save":
        with open_local_model(cfg) as f:
            return torch.load(f, weights_only=True)
        
    if cfg["write-method"]["name"] == "writeall" or cfg["write-method"]["name"] == "write-partial":
        return preload_model_bytes(cfg)


def preload_model_bytes(cfg):
    with open_local_model(cfg) as f:
        return f.read()


def get_blob_url(blob_cfg, name):
    return f'https://{blob_cfg["account"]}.blob.core.windows.net/{blob_cfg["container"]}/write-output/{name}'


def retrieve_sas_token(cfg):
    return os.environ.get(cfg["blob"]["sas-env-var"], None)


if __name__ == "__main__":
    write_perf()