import sys
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.identity import DefaultAzureCredential
import os

if len(sys.argv) < 2:
    print("Usage: upload_trirank.py <tenant_id>")
    sys.exit(1)

tenant_id = sys.argv[1]

storage_account_name = "stcourserecom"
account_url = f"https://{storage_account_name}.blob.core.windows.net"
default_credential = DefaultAzureCredential()
blob_service_client = BlobServiceClient(account_url, credential=default_credential)

container_name = "model"
container_client = blob_service_client.get_container_client(container_name)
if not container_client.exists():
    container_client.create_container()

local_path = "./save_dir/TriRank"

local_model_file_name = f"{tenant_id}.model.pkl"
upload_model_file_path = os.path.join(local_path, local_model_file_name)
blob_client = blob_service_client.get_blob_client(container=container_name, blob=local_model_file_name)
print("\nUploading to Azure Storage as blob:\n\t" + local_model_file_name)
with open(file=upload_model_file_path, mode="rb") as data:
    blob_client.upload_blob(data, overwrite=True)

local_trainset_file_name = f"{tenant_id}.model.pkl.trainset"
upload_trainset_file_path = os.path.join(local_path, local_trainset_file_name)
blob_client = blob_service_client.get_blob_client(container=container_name, blob=local_trainset_file_name)
print("\nUploading to Azure Storage as blob:\n\t" + local_trainset_file_name)
with open(file=upload_trainset_file_path, mode="rb") as data:
    blob_client.upload_blob(data, overwrite=True)
