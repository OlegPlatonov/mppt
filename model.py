import torch
from torch import nn
from transformers import BertModel
from transformers.models.bert.configuration_bert import BertConfig


class MPPT(nn.Module):
    def __init__(self, input_dim, num_targets, num_layers, hidden_dim, num_heads, hidden_dim_multiplier, dropout,
                 attn_dropout):
        super().__init__()

        transformer_config = BertConfig(
            num_hidden_layers=num_layers,
            hidden_size=hidden_dim,
            num_attention_heads=num_heads,
            intermediate_size=int(hidden_dim * hidden_dim_multiplier),
            hidden_act='gelu',
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=attn_dropout,
            vocab_size=1,
            max_position_embeddings=1,
            type_vocab_size=1
        )

        self.input_mlp = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=int(hidden_dim * hidden_dim_multiplier)),
            nn.GELU(),
            nn.LayerNorm(int(hidden_dim * hidden_dim_multiplier)),
            nn.Linear(in_features=int(hidden_dim * hidden_dim_multiplier), out_features=hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        self.transformer = BertModel(transformer_config, add_pooling_layer=False)

        self.output_linear = nn.Linear(in_features=hidden_dim, out_features=num_targets)

    def forward(self, x, attn_mask):
        transformer_inputs = self.input_mlp(x)

        position_ids = torch.zeros_like(attn_mask, dtype=int)
        token_type_ids = torch.zeros_like(attn_mask, dtype=int)

        transformer_outputs = self.transformer(inputs_embeds=transformer_inputs, attention_mask=attn_mask,
                                               position_ids=position_ids, token_type_ids=token_type_ids)
        molecule_embeddings = transformer_outputs['last_hidden_state'][:, 0]

        preds = self.output_linear(molecule_embeddings).squeeze(1)

        return preds
