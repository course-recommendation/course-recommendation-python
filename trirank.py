# Copyright 2018 The Cornac Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""TriRank: Review-aware Explainable Recommendation by Modeling Aspects"""

import sys
import cornac
from cornac.data import SentimentModality
from cornac.data import Reader
from cornac.eval_methods import RatioSplit
import os
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.identity import DefaultAzureCredential

if len(sys.argv) < 2:
    print("Usage: trirank.py <tenant_id>")
    sys.exit(1)

tenant_id = sys.argv[1]

storage_account_name = "stcourserecom"
container_name = "dataset"
account_url = f"https://{storage_account_name}.blob.core.windows.net"
default_credential = DefaultAzureCredential()
blob_service_client = BlobServiceClient(account_url, credential=default_credential)
container_client = blob_service_client.get_container_client(container=container_name)

RATING_PATH = f"./dataset/trirank/{tenant_id}.rating.txt"
SENTIMENT_PATH = f"./dataset/trirank/{tenant_id}.sentiment.txt"

os.makedirs(os.path.dirname(RATING_PATH), exist_ok=True)

print("\nDownloading blob to " + RATING_PATH)
with open(file=RATING_PATH, mode="wb") as download_file:
    download_file.write(container_client.download_blob(f"{tenant_id}.rating.txt").readall())

print("\nDownloading blob to " + SENTIMENT_PATH)
with open(file=SENTIMENT_PATH, mode="wb") as download_file:
    download_file.write(container_client.download_blob(f"{tenant_id}.sentiment.txt").readall())

reader = Reader()
rating = reader.read(RATING_PATH, fmt='UIR', sep=',')
sentiment = reader.read(SENTIMENT_PATH, fmt='UITup', sep=',', tup_sep=':')

md = SentimentModality(data=sentiment)

eval_method = RatioSplit(
    data=rating,
    test_size=0.2,
    exclude_unknowns=True,
    verbose=True,
    sentiment=md,
    seed=123,
)

trirank = cornac.models.TriRank(
    verbose=True,
    seed=123,
)

trirank.fit(eval_method.train_set)

trirank.save(save_dir="save_dir", save_trainset=True)

save_dir = "save_dir/TriRank"
new_name = f"{tenant_id}.model"

for filename in os.listdir(save_dir):
    base = filename.replace(".pkl.trainset", "").replace(".pkl.meta", "").replace(".pkl", "")
    if base != new_name and base != filename:
        new_filename = filename.replace(base, new_name)
        os.rename(os.path.join(save_dir, filename), os.path.join(save_dir, new_filename))
        print(f"Renamed: {filename} -> {new_filename}")
