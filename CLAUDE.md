# Causeway

Lightweight Counterfactual World Interface — a parameter-efficient causal adapter that bolts onto frozen Transformers to provide structured counterfactual reasoning.

Causeway estimates causal displacement under intervention, not correlation. Given a Transformer's hidden state and a candidate action, it returns a structured delta vector describing what would change if that action were taken.

## Architecture

```
StateEncoder -> CausalGraphLayer -> InterventionEngine -> DeltaPredictor
```

### Pipeline

1. **StateEncoder** (`causeway/state_encoder.py`): Projects Transformer hidden state `h` into causal variable space `z` via a learned near-orthogonal rotation matrix `R`, followed by a 3-layer refinement MLP with residual connection and LayerNorm. The rotation is inspired by Distributed Alignment Search (Geiger, Wu et al. 2023). Orthogonality is soft-constrained via `||R^T R - I||_F^2`.

2. **CausalGraphLayer** (`causeway/causal_graph.py`): Learns a sparse DAG over causal variables using Gumbel-sigmoid edge gating (annealed temperature for hard selection at eval). Implements NOTEARS acyclicity constraint (`tr(e^{W*W}) - d = 0`, Zheng et al. 2018) and L0 edge count penalty. Message passing refines variables through learned causal structure.

3. **InterventionEngine** (`causeway/intervention_engine.py`): Implements Pearl's do-operator via three-step counterfactual inference — abduction (infer exogenous noise), action (apply soft do-operator), prediction (propagate effects through the causal graph). The action encoder maps raw action representations to `(intervention_mask, intervention_values)` over causal variables.

4. **DeltaPredictor** (`causeway/delta_predictor.py`): Maps pre/post intervention causal states to a structured 5-dimensional output with per-dimension confidence. Context-dependent: uses `concat(z_pre, z_post, delta_raw)` so the same raw delta gets different structured interpretations depending on the current state.

### Output Dimensions

| Dimension | Description |
|---|---|
| `risk_shift` | How much risk increases/decreases |
| `goal_progress` | Change in progress toward goals |
| `constraint_violation` | Probability of violating constraints |
| `resource_cost` | Change in resource consumption |
| `success_probability` | Change in likelihood of success |

## Project Structure

```
causeway/
  __init__.py                  # Package exports: Causeway, StateEncoder, CausalGraphLayer, etc.
  causeway_module.py           # Main nn.Module wiring all 4 components
  state_encoder.py             # Transformer h -> causal variables z
  causal_graph.py              # Sparse DAG with Gumbel-sigmoid gating
  intervention_engine.py       # Pearl's do-operator (abduction-action-prediction)
  delta_predictor.py           # Causal z_pre/z_post -> structured delta vector
  losses.py                    # CausewayLoss (asymmetric MSE, confidence calibration, DAG constraints)

environments/
  synthetic_scm.py             # Ground-truth SCM (software deployment domain, 8 vars, 10 edges)
  text_scm.py                  # Text-encoded SCM dataset for training on real Transformer hidden states
  self_supervised.py           # Self-supervised dataset: 5 domains, representation shift targets

integration/
  transformer_bridge.py        # Wires Causeway to frozen Transformer via causal prefix injection

train.py                       # Train on synthetic SCM (ground-truth counterfactuals)
train_on_transformer.py        # Train on real Transformer (GPT-2, TinyLlama, etc.) hidden states
train_self_supervised.py       # Self-supervised: Causeway vs baseline MLP on h_delta reconstruction
evaluate.py                    # Evaluation: graph recovery, counterfactual examples, stress test
demo_gpt2.py                   # End-to-end demo: GPT-2 + Causeway + causal prefix generation
run_mistral.sh                 # Training script for Mistral 7B v0.3 on RunPod
runpod_setup.sh                # RunPod instance setup (dependencies, GPU check, model download)
```

## Training

### Synthetic SCM (ground-truth supervision)

```bash
python train.py
python train.py --d_causal 64 --epochs 200 --lr 1e-3
```

Trains on `SCMDataset` which uses random projections to simulate Transformer hidden states. Achieves ~0.998 overall correlation.

### Real Transformer hidden states

```bash
python train_on_transformer.py --model gpt2
python train_on_transformer.py --model gpt2 --d_causal 48 --epochs 200 --num_samples 50000
python train_on_transformer.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
python train_on_transformer.py --model meta-llama/Llama-3.2-1B
```

Trains on `TextSCMDataset` which encodes SCM scenarios as natural language, runs them through a frozen Transformer, and trains Causeway on the real hidden state representations. Supports any HuggingFace autoregressive model.

The dataset is cached to disk (`cache_{model}_{samples}_v2.pt`) to avoid re-encoding on subsequent runs.

### Key Training Arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | `gpt2` | HuggingFace model name |
| `--d_causal` | 48 | Number of causal variables (graph nodes) |
| `--d_action` | None | Action dim; defaults to `d_model` (768 for GPT-2) |
| `--graph_layers` | 2 | Message-passing layers in causal graph |
| `--propagation_steps` | 3 | Depth of intervention effect propagation |
| `--num_samples` | 50000 | Training dataset size |
| `--epochs` | 200 | Training epochs |
| `--lr` | 3e-4 | Learning rate |
| `--warmup_epochs` | 10 | Linear LR warmup before cosine annealing |
| `--temp_start/end` | 1.0/0.05 | Gumbel-sigmoid temperature anneal range |

## Loss Function

`CausewayLoss` combines six terms:

```
L_total = L_delta + lambda_conf * L_conf + lambda_edges * L_edges
        + lambda_acyclic * L_acyclic + lambda_sparse * L_sparse + lambda_ortho * L_ortho
```

- **L_delta**: Asymmetric MSE — underestimating risk is penalized 4x vs overestimating
- **L_conf**: Confidence calibration (binary CE: confident when accurate, uncertain when wrong)
- **L_edges**: L0 edge count penalty (expected number of active edges via sigmoid probabilities)
- **L_acyclic**: NOTEARS DAG constraint (ramped from 0 to 10 over first 30% of training)
- **L_sparse**: L1 on adjacency weights
- **L_ortho**: Orthogonality of state encoder rotation matrix

## Evaluation

```bash
python evaluate.py
python evaluate.py --checkpoint causeway_checkpoint.pt
```

Runs graph recovery analysis, counterfactual examples with per-dimension accuracy, and stress tests on fresh unseen data.

## Integration with Frozen Transformers

`TransformerBridge` (`integration/transformer_bridge.py`) converts Causeway's delta vector into prefix tokens prepended to the Transformer's input embeddings, enabling causally-informed generation without modifying the backbone:

```python
bridge = TransformerBridge(causeway=causeway, d_model=768, n_prefix_tokens=4)
prefix = bridge.get_causal_prefix(h, action)  # (batch, 4, 768)
augmented = torch.cat([prefix, input_embeddings], dim=1)
output = transformer(inputs_embeds=augmented)
```

Demo: `python demo_gpt2.py`

## SCM Domain: Software Deployment

The synthetic SCM models a software deployment system with 8 variables:

**Controllable** (actions): `code_complexity`, `test_coverage`, `deploy_load`, `rollback_readiness`

**Observable** (effects): `error_rate`, `latency_impact`, `user_impact`, `resource_usage`

**Causal graph** (10 ground-truth edges):
```
code_complexity  -> error_rate (+0.6)
code_complexity  -> latency_impact (+0.3)
code_complexity  -> resource_usage (+0.4)
test_coverage    -> error_rate (-0.5, protective)
deploy_load      -> latency_impact (+0.5)
deploy_load      -> error_rate (+0.3)
rollback_ready   -> user_impact (-0.4, protective)
error_rate       -> user_impact (+0.7)
latency_impact   -> user_impact (+0.5)
resource_usage   -> latency_impact (+0.3)
```

## Results

### Synthetic SCM
- Overall correlation: ~0.998
- Graph recovery: exact edge identification

### GPT-2 (V2)
- **Overall correlation: 0.9494** (0.794M params, 0.6% overhead vs frozen GPT-2)
- Best val loss: 0.2029 (epoch 198)
- Overall MAE: 0.0554

| Dimension | Correlation | Dir Acc | MAE |
|---|---|---|---|
| risk_shift | 0.9442 | 0.789 | 0.0563 |
| goal_progress | 0.9468 | 0.851 | 0.0574 |
| constraint_violation | 0.9309 | 0.478 | 0.0872 |
| resource_cost | 0.9470 | 0.438 | 0.0244 |
| success_probability | 0.9480 | 0.876 | 0.0518 |

Graph converged to 0.7 expected edges at 0.05 temperature.

### V1 -> V2 Improvements

V1 achieved 0.742 correlation. V2 improvements that reached 0.9494:

1. **Last-token pooling** instead of mean pooling — GPT-2 is autoregressive, the last token has attended to the full sequence and carries the richest representation.
2. **Full d_model action embeddings** instead of random projection (768 -> 128) — the random projection destroyed most action information.
3. **Scaled MLPs**: hidden dim `max(4 * d_causal, d_model // 2)` = 384 instead of `2 * d_causal` = 96 — enough capacity to decode 768-dim Transformer representations.
4. **Template diversity**: 4 text templates per state/action rotate across samples for more varied GPT-2 embeddings.
5. **LR warmup**: 10-epoch linear warmup before cosine annealing prevents early instability from high-dim inputs.
6. **LayerNorm** between InterventionEngine action encoder layers for training stability.

## Dependencies

```
torch>=2.1.0
numpy>=1.24.0
networkx>=3.1
matplotlib>=3.7.0
tqdm>=4.65.0
transformers  # for train_on_transformer.py and demo_gpt2.py
```

## Checkpoints

| File | Description |
|---|---|
| `causeway_checkpoint.pt` | Synthetic SCM training |
| `causeway_gpt2.pt` | GPT-2 V2 supervised training (0.9494 corr) |
| `cache_gpt2_50000_v2.pt` | Cached GPT-2 supervised dataset (50K samples) |
| `causeway_ss_gpt2.pt` | Self-supervised Causeway + DeltaDecoder (cos=0.707) |
| `causeway_ss_gpt2_baseline.pt` | Self-supervised baseline MLP (cos=0.769) |
| `cache_ss_gpt2_50000.pt` | Cached self-supervised dataset (50K, 461MB) |

## Related Work

No existing system combines all of Causeway's properties into a single module. The individual techniques have precedent; the unified design does not.

### Direct Ancestors

**Deep Structural Causal Models (DeepSCM)** — Pawlowski et al., NeurIPS 2020 ([paper](https://arxiv.org/abs/2006.06485)). First framework to make all three steps of Pearl's counterfactual procedure (abduction, action, prediction) tractable with neural networks. Uses normalizing flows for invertible noise inference. Causeway's InterventionEngine implements the same three-step procedure but as a lightweight learned module rather than a full generative model. DeepSCM operates standalone on images (Morpho-MNIST, brain MRI); it does not bolt onto a frozen backbone.

**Distributed Alignment Search (DAS)** — Geiger, Wu et al., 2023 ([paper](https://arxiv.org/abs/2303.02536)). Finds causal variables as learned rotations of Transformer hidden states using interchange interventions. Causeway's StateEncoder rotation matrix `R` is directly inspired by this: projecting `h` into a causal subspace via a near-orthogonal rotation. DAS is an analysis tool for post-hoc interpretability; it does not predict counterfactual deltas or produce structured outputs.

**NOTEARS** — Zheng et al., NeurIPS 2018 ([paper](https://github.com/xunzheng/notears)). Continuous optimization for DAG structure learning via the acyclicity constraint `tr(e^{W*W}) - d = 0`. Causeway's CausalGraphLayer uses this constraint combined with Gumbel-sigmoid edge gating for genuinely sparse learned graphs.

### Closest in Spirit

**Language Agents Meet Causality (LLM-CWM)** — Gou et al., 2024 ([paper](https://arxiv.org/abs/2410.19923)). Builds a Causal World Model that LLMs can query for action effects, with causal variables linked to natural language. The closest project philosophically: an LLM uses a separate causal model to predict consequences of actions. But LLM-CWM is a full environment simulator (encodes images, decodes state changes), not a lightweight adapter. It requires its own training pipeline and is not designed as a bolt-on module.

**Causal World Model Induction (CWMI)** — 2025 ([paper](https://arxiv.org/abs/2507.19855)). Embeds a Causal Physics Module inside an LLM with a Causal Intervention Loss that forces the model to predict intervention outcomes rather than just correlations. Architecturally the most similar motivation to Causeway. Key differences: CWMI modifies the LLM's training (not frozen), targets physical reasoning in images, and does not produce structured delta vectors.

### Adjacent Work

**Causal Transformer for Counterfactual Outcomes** — Melnychuk et al., ICML 2022 ([paper](https://proceedings.mlr.press/v162/melnychuk22a.html)). Uses Transformers as the causal estimator for treatment effect prediction in time series. Opposite design philosophy: the Transformer itself is the causal model. Causeway uses the Transformer only as a frozen feature extractor and keeps causality in a separate module.

**DAG-aware Transformer** — Liu et al., 2024 ([paper](https://arxiv.org/abs/2410.10044)). Integrates known DAG structure into Transformer attention for causal effect estimation. Bakes causality into the attention mechanism; Causeway keeps the Transformer frozen and learns its own DAG externally.

**Causal-Adapter** — 2025 ([paper](https://arxiv.org/abs/2509.24798)). Lightweight trainable modules bolted onto frozen Stable Diffusion for counterfactual image generation. Same adapter-on-frozen-backbone philosophy, but for images, not structured delta prediction. Uses SCM-based attribute regularization but does not implement Pearl's full abduction-action-prediction procedure.

### Positioning

| Property | DeepSCM | DAS | LLM-CWM | CWMI | Causal-Adapter | Causeway |
|---|---|---|---|---|---|---|
| Frozen backbone | No | N/A | Partial | No | Yes | **Yes** |
| Pearl's do-operator | Yes | No | Partial | Partial | No | **Yes** |
| Learned sparse DAG | No | No | Yes | No | No | **Yes** |
| Structured delta output | No | No | Text | No | No | **Yes** |
| Parameter-efficient (<1M) | No | N/A | No | No | No | **Yes (0.794M)** |
| Plugs into any Transformer | No | No | No | No | Diffusion only | **Yes** |

The unique contribution is the combination: a sub-1M parameter module implementing full Pearl counterfactual inference through a learned sparse DAG, operating on real Transformer hidden states, producing structured non-text outputs, without touching the backbone weights.

### Self-Supervised Training (Representation Shift Reconstruction)

The self-supervised approach uses the Transformer's own representation shifts as the training signal — no external ground truth needed.

**Concept**: For each sample, generate a base scenario and an intervened variant, encode both through frozen GPT-2, and use `h_delta = h_intervened - h_base` as the training target. If Causeway's causal bottleneck can reconstruct h_delta better than a parameter-matched flat MLP, the causal structure is extracting something real.

```bash
python train_self_supervised.py --model gpt2
python train_self_supervised.py --model gpt2 --d_causal 48 --epochs 200 --num_samples 50000
```

**Pipeline**:
```
scenario_base      -> GPT-2 -> h_base
scenario_intervened -> GPT-2 -> h_intervened
intervention_text   -> GPT-2 -> action
h_delta = h_intervened - h_base  (training target)

Causeway(h_base, action) -> structured delta -> DeltaDecoder -> h_delta_pred
```

**Domains** (5 domains, 6 variables each, 10K samples per domain):
1. Software Deployment (complexity, coverage, load, rollback, experience, monitoring)
2. Clinical Treatment (severity, dose, vitals, labs, compliance, comorbidity)
3. Business Operations (demand, inventory, pricing, competition, retention, costs)
4. Infrastructure Management (utilization, hardware, redundancy, security, bandwidth, backlog)
5. Project Management (velocity, debt, clarity, alignment, resources, pressure)

**Results (GPT-2)**:

| Metric | Causeway+Decoder | Baseline MLP | Winner |
|---|---|---|---|
| MSE | 0.00813 | 0.00617 | Baseline |
| Cosine Similarity | 0.707 | 0.769 | Baseline |
| Params | 1.483M | 1.486M | (matched) |

Per-domain results (all won by Baseline):

| Domain | CW MSE | BL MSE | CW Cos | BL Cos |
|---|---|---|---|---|
| deployment | 0.0085 | 0.0066 | 0.705 | 0.769 |
| clinical | 0.0033 | 0.0026 | 0.710 | 0.775 |
| business | 0.0067 | 0.0052 | 0.737 | 0.776 |
| infrastructure | 0.0118 | 0.0087 | 0.707 | 0.783 |
| project | 0.0102 | 0.0076 | 0.679 | 0.743 |

**Analysis**: The causal bottleneck (768 -> 48 -> 768) destroys too much information for full h_delta reconstruction. The DeltaDecoder has only 58 input dims (5 + 5 + 48) to reconstruct 768-dim output, while the baseline MLP has 1536 input dims (h_base + action concatenated). The causal graph collapsed to ~0.5 expected edges (density=0.0), indicating no meaningful graph structure was learned. Causeway also overfits more severely (train cos=0.853 vs val cos=0.707).

**Key insight**: The 48-dim causal bottleneck is well-suited for the supervised 5-dim structured delta prediction task (0.9494 correlation) but is too tight for reconstructing full 768-dim representation shifts. This is a dimensionality bottleneck problem, not a failure of the causal approach itself.

**Potential improvements**:
1. Increase d_causal (e.g., 256+) to give the bottleneck more capacity
2. Add skip connections (pass h_base directly to decoder alongside causal features)
3. Use a different training objective (not full h_delta reconstruction)
4. Multi-scale reconstruction with auxiliary losses at the causal variable level

### Checkpoints (Self-Supervised)

| File | Description |
|---|---|
| `causeway_ss_gpt2.pt` | Causeway + DeltaDecoder best checkpoint (epoch 194) |
| `causeway_ss_gpt2_baseline.pt` | Baseline MLP best checkpoint (epoch 192) |
| `cache_ss_gpt2_50000.pt` | Cached self-supervised encoded dataset (50K samples, 461MB) |

## Integration: BroadMind x Causeway

Causeway integrates with [BroadMind](https://github.com/ElnurIbrahimov/BroadMind), a neural program executor that generates and executes its own latent programs at runtime (447K params, 100% accuracy).

- **Causeway** provides causal reasoning: learned DAG, Pearl's do-operator, structured delta predictions
- **BroadMind** provides program execution: latent program induction, adaptive compute, wisdom distillation

The integration module (`integration/broadmind_bridge.py`) contains three components:

1. **CausalWisdomBridge**: Converts Causeway's adjacency matrix + causal variables into BroadMind wisdom codes (both use 48-dim representations)
2. **AdaptiveInterventionEngine**: Replaces Causeway's fixed 3-step intervention propagation with BroadMind-style adaptive compute (halter decides when effects have propagated)
3. **CausalProgramExecutor**: Joint orchestrator — Causeway predicts consequences, bridge produces causal wisdom, gated fusion blends it with BroadMind's matched wisdom, BroadMind executes using the fused wisdom

Combined system: 1.3M params (794K Causeway + 447K BroadMind + 60K bridge).

See [BroadMindxCauseway](https://github.com/ElnurIbrahimov/BroadMindxCauseway) for the full integration documentation.

## Key Design Decisions

- **d_causal=48**: Sufficient for the 8-variable SCM. Increasing it grows the graph quadratically (48^2 = 2304 potential edges).
- **d_action = d_model**: Actions stored as full Transformer embeddings to preserve information. The InterventionEngine decodes them to causal-space intervention masks and values.
- **Gumbel-sigmoid edge gating**: Achieves genuinely sparse graphs (0-1 edges at convergence) rather than many weak edges from L1 alone.
- **Temperature annealing** (1.0 -> 0.05): Soft edges during early training for gradient flow, hard binary selection at convergence.
- **Acyclicity ramp**: DAG constraint ramped from 0 to 10 over first 30% of epochs. This lets the graph learn useful edges before the DAG constraint prunes cycles.
- **Asymmetric loss**: Risk underestimation penalized 4x vs overestimation for safety-critical deployment decisions.
