import datetime

import docker
import pytest
from azure.storage.blob import BlobServiceClient

from adlfs.tests.constants import ACCOUNT_NAME, KEY, URL

data = b"0123456789"
metadata = {"meta": "data"}


def pytest_addoption(parser):
    parser.addoption(
        "--host",
        action="store",
        default="127.0.0.1:10000",
        help="Host running azurite.",
    )


@pytest.fixture(scope="session")
def host(request):
    print("host:", request.config.getoption("--host"))
    return request.config.getoption("--host")


@pytest.fixture(scope="function")
def storage(host):
    """
    Create blob using azurite.
    """

    conn_str = f"DefaultEndpointsProtocol=http;AccountName={ACCOUNT_NAME};AccountKey={KEY};BlobEndpoint={URL}/{ACCOUNT_NAME};"  # NOQA

    bbs = BlobServiceClient.from_connection_string(conn_str=conn_str)
    if "data" not in [c["name"] for c in bbs.list_containers()]:
        bbs.create_container("data")
    container_client = bbs.get_container_client(container="data")
    bbs.insert_time = datetime.datetime.now(tz=datetime.timezone.utc).replace(
        microsecond=0
    )
    container_client.upload_blob("top_file.txt", data)
    container_client.upload_blob("root/rfile.txt", data)
    container_client.upload_blob("root/a/file.txt", data)
    container_client.upload_blob("root/a1/file1.txt", data)
    container_client.upload_blob("root/b/file.txt", data)
    container_client.upload_blob("root/c/file1.txt", data)
    container_client.upload_blob("root/c/file2.txt", data)
    container_client.upload_blob(
        "root/d/file_with_metadata.txt", data, metadata=metadata
    )
    container_client.upload_blob("root/e+f/file1.txt", data)
    container_client.upload_blob("root/e+f/file2.txt", data)
    yield bbs

    bbs.delete_container("data")


@pytest.fixture(scope="session", autouse=True)
def spawn_azurite():
    print("Starting azurite docker container")
    client = docker.from_env()
    azurite = client.containers.run(
        "mcr.microsoft.com/azure-storage/azurite",
        "azurite-blob --loose --blobHost 0.0.0.0 --skipApiVersionCheck",
        detach=True,
        ports={"10000": "10000"},
    )
    print("Successfully created azurite container...")
    yield azurite
    print("Teardown azurite docker container")
    azurite.stop()
