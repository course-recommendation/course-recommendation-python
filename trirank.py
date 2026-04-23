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

import cornac
from cornac.datasets import amazon_toy
from cornac.data import SentimentModality
from cornac.eval_methods import RatioSplit
import os


# Load rating and sentiment information
rating = amazon_toy.load_feedback()
sentiment = amazon_toy.load_sentiment()

# Instantiate a SentimentModality, it makes it convenient to work with sentiment information
md = SentimentModality(data=sentiment)

# Define an evaluation method to split feedback into train and test sets
eval_method = RatioSplit(
    data=rating,
    test_size=0.2,
    exclude_unknowns=True,
    verbose=True,
    sentiment=md,
    seed=123,
)

# Instantiate the model
trirank = cornac.models.TriRank(
    verbose=True,
    seed=123,
)

trirank.fit(eval_method.train_set)

trirank.save(save_dir="save_dir", save_trainset=True)

save_dir = "save_dir/TriRank"
new_name = "model"

for filename in os.listdir(save_dir):
    # Strip known extensions to get the base name
    base = filename.replace(".pkl.trainset", "").replace(".pkl.meta", "").replace(".pkl", "")
    if base != new_name and base != filename:  # it's a generated timestamp name
        new_filename = filename.replace(base, new_name)
        os.rename(os.path.join(save_dir, filename), os.path.join(save_dir, new_filename))
        print(f"Renamed: {filename} -> {new_filename}")