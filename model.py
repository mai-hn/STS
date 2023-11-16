import paddle
import paddle.nn as nn
from paddle.nn.functional import cosine_similarity

class SentenceTransformer(nn.Layer):
    def __init__(self, pretrained_model, dropout=None):
        super().__init__()
        self.ptm = pretrained_model
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)

        # MLP Module for feature
        self.feature_mlp = nn.Sequential(
            nn.Linear(1024, self.ptm.config["hidden_size"]),
            nn.ReLU(),
            nn.Linear(self.ptm.config["hidden_size"], self.ptm.config["hidden_size"])
        )

    def encode(self, input_ids, token_type_ids=None, attention_mask=None):

        embedding, _ = self.ptm(input_ids, token_type_ids, attention_mask=attention_mask)
        embedding = self.dropout(embedding)

        attention_mask = paddle.unsqueeze((input_ids != self.ptm.pad_token_id).astype(self.ptm.pooler.dense.weight.dtype), axis=2)
        # Apply attention mask to the embeddings
        embedding = embedding * attention_mask
        sum_embedding = paddle.sum(embedding, axis=1)
        sum_mask = paddle.sum(attention_mask, axis=1)
        mean_embedding = sum_embedding / sum_mask
        return mean_embedding



    def forward(
        self,
        query_input_ids,
        query_token_type_ids=None,
        feature=None,
        title_embeddings=None,
        isencode=False
    ):
        if isencode:
            return self.encode(query_input_ids, query_token_type_ids)

        query_embedding = self.encode(
            query_input_ids, query_token_type_ids
        )

        # Process feature through the MLP
        feature_transformed = self.feature_mlp(feature)
        feature_transformed = paddle.reshape(feature_transformed, shape=[-1, self.ptm.config["hidden_size"]])

        # Average the query embedding and feature representation
        combined_mean = (query_embedding + feature_transformed) / 2


        # Expand the query embedding to the same size as the title embeddings
        combined_mean_expanded = paddle.unsqueeze(combined_mean, axis=1)  # 形状变为 (32, 1, 768)
        title_embeddings_expanded = paddle.unsqueeze(title_embeddings, axis=0)  # 形状变为 (1, 191, 768)

        # 使用 paddle 的 cosine_similarity 计算余弦相似度
        logits = paddle.nn.functional.cosine_similarity(combined_mean_expanded, title_embeddings_expanded, axis=2)
        logits = paddle.squeeze(logits, axis=[1])
        return logits

