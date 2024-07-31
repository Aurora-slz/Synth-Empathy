import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from sentence_transformers import SentenceTransformer, util
import json
import random
import matplotlib.pyplot as plt

from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

# from anomalib.models.components.dimensionality_reduction import SparseRandomProjection


class KCenterGreedy:
    """Implements k-center-greedy method.

    Args:
        embedding (Tensor): Embedding vector extracted from a CNN
        sampling_ratio (float): Ratio to choose coreset size from the embedding size.

    Example:
        >>> embedding.shape
        torch.Size([219520, 1536])
        >>> sampler = KCenterGreedy(embedding=embedding)
        >>> sampled_idxs = sampler.select_coreset_idxs()
        >>> coreset = embedding[sampled_idxs]
        >>> coreset.shape
        torch.Size([219, 1536])
    """

    def __init__(self, embedding: Tensor, sampling_ratio: float) -> None:
        self.embedding = embedding
        self.coreset_size = int(embedding.shape[0] * sampling_ratio)
        # self.model = SparseRandomProjection(eps=0.9)

        self.features: Tensor
        self.min_distances: Tensor = None
        self.n_observations = self.embedding.shape[0]

    def reset_distances(self) -> None:
        """Reset minimum distances."""
        self.min_distances = None

    def update_distances(self, cluster_centers: List[int]) -> None:
        """Update min distances given cluster centers.

        Args:
            cluster_centers (List[int]): indices of cluster centers
        """

        if cluster_centers:
            centers = self.features[cluster_centers]

            distance = F.pairwise_distance(self.features, centers, p=2).reshape(-1, 1)

            if self.min_distances is None:
                self.min_distances = distance
            else:
                self.min_distances = torch.minimum(self.min_distances, distance)

    def get_new_idx(self) -> int:
        """Get index value of a sample.

        Based on minimum distance of the cluster

        Returns:
            int: Sample index
        """

        if isinstance(self.min_distances, Tensor):
            idx = int(torch.argmax(self.min_distances).item())
        else:
            raise ValueError(f"self.min_distances must be of type Tensor. Got {type(self.min_distances)}")

        return idx

    def select_coreset_idxs(self, selected_idxs: Optional[List[int]] = None) -> List[int]:
        """Greedily form a coreset to minimize the maximum distance of a cluster.

        Args:
            selected_idxs: index of samples already selected. Defaults to an empty set.

        Returns:
          indices of samples selected to minimize distance to cluster centers
        """

        if selected_idxs is None:
            selected_idxs = []

        if self.embedding.ndim == 2:
            # self.model.fit(self.embedding)
            # self.features = self.model.transform(self.embedding)
            
            self.features = self.embedding
            self.reset_distances()
        else:
            self.features = self.embedding.reshape(self.embedding.shape[0], -1)
            self.update_distances(cluster_centers=selected_idxs)

        selected_coreset_idxs: List[int] = []
        idx = int(torch.randint(high=self.n_observations, size=(1,)).item())
        cnt = 0
        for _ in range(self.coreset_size):
            cnt += 1
            if(cnt % 1000 == 0):
                print(cnt)
            self.update_distances(cluster_centers=[idx])
            idx = self.get_new_idx()
            if idx in selected_idxs:
                raise ValueError("New indices should not be in selected indices.")
            self.min_distances[idx] = 0
            selected_coreset_idxs.append(idx)

        return selected_coreset_idxs

    def sample_coreset(self, selected_idxs: Optional[List[int]] = None) -> Tensor:
        """Select coreset from the embedding.

        Args:
            selected_idxs: index of samples already selected. Defaults to an empty set.

        Returns:
            Tensor: Output coreset

        Example:
            >>> embedding.shape
            torch.Size([219520, 1536])
            >>> sampler = KCenterGreedy(...)
            >>> coreset = sampler.sample_coreset()
            >>> coreset.shape
            torch.Size([219, 1536])
        """

        idxs = self.select_coreset_idxs(selected_idxs)
        coreset = self.embedding[idxs]

        return coreset
    




def load_file(load_path):
    with open(load_path, 'r', encoding='utf-8') as f1:
        data = json.load(f1)
        print(data[0])
    return data

def save_file(data, save_path):
    with open(save_path, 'w', encoding='utf-8') as f1:
        f1.write(json.dumps(data))





model = SentenceTransformer("gte-Qwen2-7B-instruct", trust_remote_code=True)
model.max_seq_length = 8192

if __name__ == '__main__':

    load_path = "/select_data/filter_embedscore60_llama3_sft_base_origin_checkpoint-3200_score_expandOriginData_70b_race3.json"
    test_data = load_file(load_path)
    conversations = []
    for i in range(0, len(test_data)):
        conversations.append(test_data[i]['instruction']+test_data[i]['output'])

    conversation_embeddings = model.encode(conversations, prompt_name="query")


    embedding = torch.tensor(conversation_embeddings)
    print(embedding.shape)

    sampler = KCenterGreedy(embedding=embedding, sampling_ratio=0.9)
    sampled_idxs = sampler.select_coreset_idxs()
    coreset = embedding[sampled_idxs]
    # print('select: ', coreset)
    print(coreset.shape)
    print('sampled_idxs: ', sampled_idxs) 

    with open("/select_data/kcentergreedy0.9_idx_filter_embedscore60_llama3_sft_base_origin_checkpoint-3200_score_expandOriginData_70b_race3.json", 'w') as file:
        json.dump(sampled_idxs, file)
