# TransformerRecModel Architecture and Parameter Calculation

## Architecture:
```text
Input (per user batch)
  input_ids: [B, L]  (movie item_idx sequence, left padded with 0)
      |
      |-- Item ID embedding table
      |     item_emb(input_ids) -> [B, L, d_model]
      |
      |-- Frozen side feature buffer (precomputed)
      |     item_side_features[input_ids] -> [B, L, side_dim]
      |     side_dim = title_dim + year_dim + genre_dim = 64 + 1 + 16 = 81
      |
      |-- Feature fusion
      |     concat([item_emb, side_feat], dim=-1) -> [B, L, d_model + side_dim]
      |     Linear(d_model + side_dim -> d_model) -> [B, L, d_model]
      |
      |-- Positional embedding
      |     pos_emb(pos_ids) -> [B, L, d_model]
      |
      |-- Add + Dropout
      |     x = fused_item_repr + pos_emb -> [B, L, d_model]
      |
      |-- Transformer Encoder (n_layers=2, MHA + FFN)
      |     src_key_padding_mask from input_ids==0
      |     h -> [B, L, d_model]
      |
      |-- Final LayerNorm
      |     h_norm -> [B, L, d_model]
      |
      |-- Last valid timestep pooling
      |     seq_repr -> [B, d_model]
      |
      |-- Scoring head (dot product)
            score_items(seq_repr, candidate_item_ids)
            candidate_item_repr uses same fusion path as above
            output scores: [B, N_candidates]

Training objective:
  BPR loss on (positive next item) vs (sampled negative items)
  loss = -log(sigmoid(score_pos - score_neg))

Evaluation:
  Sampled ranking metrics (HitRate@K, Recall@K, Precision@K, NDCG@K)
```


This document provides the exact parameter count for the
`TransformerRecModel` using the following configuration:

-   Number of movies (`num_items`) = **14093**
-   Maximum sequence length (`max_len`) = **50**
-   Model dimension (`d_model`) = **128**
-   Number of Transformer layers (`n_layers`) = **2**
-   Feedforward dimension (`dim_ff`) = **256**
-   Side feature dimension (`side_dim`) = **81**

------------------------------------------------------------------------

# 1. Embedding Layers

## 1.1 Item ID Embedding

Embedding shape:

    (num_items + 1) × d_model
    = (14093 + 1) × 128
    = 14094 × 128
    = 1,804,032

**Parameters: 1,804,032**

------------------------------------------------------------------------

## 1.2 Position Embedding

Embedding shape:

    max_len × d_model
    = 50 × 128
    = 6,400

**Parameters: 6,400**

------------------------------------------------------------------------

### Embedding subtotal

    1,804,032 + 6,400 = 1,810,432

------------------------------------------------------------------------

# 2. Fusion Layer (MLP)

This layer fuses:

-   ID embedding: 128 dims
-   Side features: 81 dims

Input dimension:

    128 + 81 = 209

Linear layer parameters:

    209 × 128 = 26,752

**Parameters: 26,752**

------------------------------------------------------------------------

# 3. Transformer Encoder Blocks

There are **2 TransformerEncoderLayer blocks**.

Each layer contains:

-   Multi-head self-attention
-   Feedforward network
-   LayerNorm

------------------------------------------------------------------------

## 3.1 Attention parameters per layer

QKV projection:

    128 × (3 × 128) = 49,152
    bias = 384

Output projection:

    128 × 128 = 16,384
    bias = 128

Total attention per layer:

    49,152 + 384 + 16,384 + 128 = 66,048

------------------------------------------------------------------------

## 3.2 Feedforward network per layer

Layer 1:

    128 × 256 = 32,768
    bias = 256

Layer 2:

    256 × 128 = 32,768
    bias = 128

Total FFN per layer:

    32,768 + 256 + 32,768 + 128 = 65,920

------------------------------------------------------------------------

## 3.3 LayerNorm per layer

Two LayerNorm layers:

    2 × (128 + 128) = 512

------------------------------------------------------------------------

## 3.4 Total per Transformer layer

    66,048 + 65,920 + 512 = 132,480

Two layers:

    132,480 × 2 = 264,960

**Parameters: 264,960**

------------------------------------------------------------------------

# 4. Final LayerNorm

    128 + 128 = 256

**Parameters: 256**

------------------------------------------------------------------------

# 5. Total Parameter Count

Summing all components:

    Embedding layers:        1,810,432
    Fusion layer:               26,752
    Transformer blocks:        264,960
    Final LayerNorm:              256
    -----------------------------------
    TOTAL:                   2,102,400

------------------------------------------------------------------------

# Final Result

## Total Trainable Parameters

    2,102,400 parameters
    ≈ 2.10 Million parameters

------------------------------------------------------------------------

# Parameter Breakdown Table

  Component            Parameters      Percentage
  -------------------- --------------- ------------
  Item embedding       1,804,032       85.8%
  Position embedding   6,400           0.3%
  Fusion layer         26,752          1.3%
  Transformer blocks   264,960         12.6%
  Final LayerNorm      256             \~0%
  **TOTAL**            **2,102,400**   **100%**

------------------------------------------------------------------------

# Notes

-   Side features (`item_side_features`) are registered as a buffer →
    **0 trainable parameters**
-   Most parameters reside in the item embedding table
-   Transformer portion is relatively lightweight compared to embeddings

------------------------------------------------------------------------
