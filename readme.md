## Implementation code and dataset for manuscript "VisionTraj: A Noise-Robust Trajectory Recovery Framework based on Large-scale Camera Network"

### TL;DR

#### Requirement
```bash
pip install requirements.txt
```

In addition, we modify the torch 1.9.1 offical code for the easy implemention of the attention-based soft-denosing.

- change function `nn.functional._scaled_dot_product_attention` to

```python
def _scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    multiply_attn: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    r"""
    Computes scaled dot product attention on query, key and value tensors, using
    an optional attention mask if passed, and applying dropout if a probability
    greater than 0.0 is specified.
    Returns a tensor pair containing attended values and attention weights.

    Args:
        q, k, v: query, key and value tensors. See Shape section for shape details.
        attn_mask: optional tensor containing mask values to be added to calculated
            attention. May be 2D or 3D; see Shape section for details.
        dropout_p: dropout probability. If greater than 0.0, dropout is applied.

    Shape:
        - q: :math:`(B, Nt, E)` where B is batch size, Nt is the target sequence length,
            and E is embedding dimension.
        - key: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
            and E is embedding dimension.
        - value: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
            and E is embedding dimension.
        - attn_mask: either a 3D tensor of shape :math:`(B, Nt, Ns)` or a 2D tensor of
            shape :math:`(Nt, Ns)`.

        - Output: attention values have shape :math:`(B, Nt, E)`; attention weights
            have shape :math:`(B, Nt, Ns)`
    """
    B, Nt, E = q.shape
    q = q / math.sqrt(E)
    # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
    attn = torch.bmm(q, k.transpose(-2, -1))
    # if multiply_attn is not None:  #### add by us
    #     attn = attn * multiply_attn ## add by us
    if attn_mask is not None:
        attn += attn_mask
    attn = softmax(attn, dim=-1)
    if multiply_attn is not None:  #### add by us
        attn = attn * multiply_attn ## add by lizhisuhai
        sum_att = torch.sum(attn, dim=-1, keepdim=True) ## add by us
        attn = torch.div(attn, sum_att) ## add by us
    if dropout_p > 0.0:
        attn = dropout(attn, p=dropout_p)
    # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
    output = torch.bmm(attn, v)
    return output, attn
```

- add `multiply_attn: Optional[Tensor] = None,` in `nn.functional.multi_head_attention_forward` line 4867 

- add `multiply_attn=multiply_attn,` in `nn.functional.multi_head_attention_forward` line 4953 

- change `_scaled_dot_product_attention(q, k, v, attn_mask, dropout_p)` to `_scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, multiply_attn)` in `nn.functional.multi_head_attention_forward` line 5091 

#### Run the model with the processed data

```bash
python models/decoder_tfm_denosie_e2e_tklet_deno.py
```


### The use of `preprocess/alg_*.ipynb`

`alg_sim_graph`: get the .osm network and simplify (consolidate_intersections).  output `longhua_1.8k.pkl`

`alg_camloc2node`: map the camera to the graph node. cam_id to node_id.

`alg_mapmatching`: represent the GPS trajectory by the graph node sequence.

`alg_tklet_extract`: extract tracklet for each captured figure.

`alg_merge_opod_features`: merge the carplate text, carplate feature, and apperance feature of all the records for OPOD.

`alg_cluster_thres`: generate the training data for traj_recovery based on high and low OPOD thresholds

### Dataset
[Download]() and unzip into ./dataset/
to be finished 
### more details to be finished 
