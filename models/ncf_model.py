import torch
import torch.nn as nn

class NCF(nn.Module):
    def __init__(self, num_users, num_items, emb_dim,
                 init_item_vectors=None,   # <-- make optional
                 freeze_items=True, user_dim=None):
        super().__init__()
        user_dim = user_dim or emb_dim

        # user embedding (trainable)
        self.user_emb = nn.Embedding(num_users, user_dim)

        # item embedding: from content tower if provided, else fresh
        if init_item_vectors is not None:
            weights = torch.tensor(init_item_vectors, dtype=torch.float32)
            self.item_emb = nn.Embedding.from_pretrained(weights, freeze=freeze_items)
        else:
            self.item_emb = nn.Embedding(num_items, emb_dim)  # fresh trainable

        # concat [u, v, u*v, |u-v|]
        in_dim = user_dim + emb_dim + emb_dim + emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, user_idx, movie_idx):
        u = self.user_emb(user_idx)
        v = self.item_emb(movie_idx)
        uv = u * v
        ud = torch.abs(u - v)
        x = torch.cat([u, v, uv, ud], dim=1)
        return self.mlp(x).squeeze(1)
