通过修改qwen的config训练参数可以达到降低显存的效果
```
{
  "attn_dropout_prob": 0.2,
  "emb_dropout_prob": 0.2,
  "hidden_size": 16,
  "intermediate_size": 4,
  "initializer_range": 0.02,
  "kv_channels": 4,
  "layer_norm_epsilon": 1e-06,
  "max_position_embeddings": 1024,
  "num_attention_heads": 4,
  "num_hidden_layers": 4,
  "rotary_emb_base": 5000,
  "seq_length": 64,
  "vocab_size": 151936
}
```
词表大小没改成功，如果改了词表大小应该显存更低