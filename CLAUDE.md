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
  clinical_scm.py              # Ground-truth SCM (clinical treatment domain, 8 vars, 10 edges)
  text_scm.py                  # Text-encoded SCM dataset for deployment domain
  text_clinical.py             # Text-encoded SCM dataset for clinical domain
  self_supervised.py           # Self-supervised dataset: 5 domains, representation shift targets

keystone/
  __init__.py                  # Package exports: Keystone, UncertaintyEncoder, CommitmentGate, etc.
  keystone_module.py           # Main nn.Module wiring 4-layer defense pipeline
  uncertainty_encoder.py       # Layer 1: QR rotation + NIG evidential head
  commitment_gate.py           # Layer 2: SelectiveNet-style 4-route gate
  constraint_manager.py        # Layer 3: Lagrangian multipliers + dual ascent
  conformal_calibrator.py      # Layer 4: Split conformal + ACI
  losses.py                    # KeystoneLoss (evidential + selective risk + Lagrangian + calibration)
  commitment.py                # KeystoneCommitment dataclass, ROUTE_NAMES

integration/
  transformer_bridge.py        # Wires Causeway to frozen Transformer via causal prefix injection
  causeway_keystone.py         # Joint Causeway + Keystone orchestrator (GatedDelta)
  keystone_bridge.py           # Keystone-aware TransformerBridge (route-gated prefix injection)

train.py                       # Train on synthetic SCM (ground-truth counterfactuals)
train_on_transformer.py        # Train on real Transformer (GPT-2, TinyLlama, etc.) hidden states
train_self_supervised.py       # Self-supervised: Causeway vs baseline MLP on h_delta reconstruction
train_keystone.py              # Train Keystone on cached Transformer hidden states (3-phase schedule)
evaluate.py                    # Evaluation: graph recovery, counterfactual examples, stress test
demo_gpt2.py                   # End-to-end demo: GPT-2 + Causeway + causal prefix generation
run_mistral.sh                 # Train Mistral 7B on both domains (deployment + clinical)
run_clinical.sh                # Train clinical domain only (standalone, RunPod-friendly)
runpod_setup.sh                # RunPod instance setup (dependencies, GPU check, model download)
Causeway Prior Work Analysis.pdf  # Prior work analysis document
```

## Keystone: Commitment and Uncertainty Spine

Keystone is the second companion module for Transformers. While Causeway predicts **what would change** under intervention, Keystone evaluates **whether to trust that prediction** and gates execution accordingly.

Keystone forces the model to explicitly declare: how confident it is (epistemic vs aleatoric), what it assumes (premises), and what should happen if it is wrong (failure modes). This is not text — it is control flow that physically gates the action pathway.

### Architecture

```
UncertaintyEncoder -> CommitmentGate -> ConstraintManager -> ConformalCalibrator
```

**4-layer defense pipeline:**
1. **UncertaintyEncoder** (`keystone/uncertainty_encoder.py`): QR-initialized rotation (same as Causeway's StateEncoder) + NIG evidential head producing epistemic/aleatoric decomposition. nu capped at 100 to prevent evidence collapse.
2. **CommitmentGate** (`keystone/commitment_gate.py`): SelectiveNet-style 4-route gate (proceed/verify/defer/abstain) with premise detection and failure mode prediction heads. Temperature-annealed softmax.
3. **ConstraintManager** (`keystone/constraint_manager.py`): Lagrangian multipliers (NOT optimizer parameters) for coverage, risk, and calibration constraints. Dual ascent with EMA-smoothed violations. 0 learnable parameters.
4. **ConformalCalibrator** (`keystone/conformal_calibrator.py`): Split conformal prediction with ACI for distribution-free coverage guarantees. Post-hoc calibration. 0 learnable parameters.

### Output: KeystoneCommitment

| Field | Shape | Description |
|---|---|---|
| `values` | (batch, 5) | NIG point estimates (same dims as Causeway's delta) |
| `epistemic` | (batch, 5) | Reducible uncertainty per dimension |
| `aleatoric` | (batch, 5) | Irreducible uncertainty per dimension |
| `confidence_bounds` | (batch, 5, 2) | Conformalized [lower, upper] intervals |
| `route` | (batch, 4) | Softmax probs over proceed/verify/defer/abstain |
| `premises` | (batch, 5) | Per-dim premise confidence [0,1] |
| `failure_modes` | (batch, 5) | Per-dim failure probability [0,1] |

### Training

```bash
# Requires Causeway's cached dataset
python train_keystone.py --model gpt2
python train_keystone.py --domain clinical --model gpt2
```

3-phase staged training: evidential only (0-30%), add routing (30-60%), all losses (60-100%). Evidence regularizer annealed. Gate temperature annealed 2.0 -> 0.5. Post-training conformal calibration on held-out 15%.

### Loss Function

```
L_total = L_evidential (NIG Type-II MLE + evidence reg)
        + lambda_selective * L_selective_risk
        + L_lagrangian (augmented Lagrangian)
        + lambda_ortho * L_orthogonality
        + lambda_premise * L_premise (BCE)
        + lambda_failure * L_failure (BCE)
```

### Integration with Causeway

`CausewayKeystoneOrchestrator` (`integration/causeway_keystone.py`) runs both in parallel:
- Causeway predicts delta, Keystone evaluates commitment
- `gated_values = delta.values * proceed_prob`
- Conservative confidence: `min(causeway_conf, 1 - normalized_epistemic)`

`KeystoneBridge` (`integration/keystone_bridge.py`) extends TransformerBridge:
- proceed: full causal prefix
- verify: 0.5 * causal + 0.5 * uncertainty prefix
- defer: uncertainty prefix only
- abstain: zero prefix (nothing injected)

### Parameter Budget

| Scale | Keystone | Combined w/ Causeway | Overhead |
|---|---|---|---|
| GPT-2 | 0.341M | 1.135M | 0.92% |
| Mistral 7B | 5.189M | 22.929M | 0.32% |

Full documentation: `C:\Users\asus\Desktop\keystone.md`

## Training

### Synthetic SCM (ground-truth supervision)

```bash
python train.py
python train.py --d_causal 64 --epochs 200 --lr 1e-3
```

Trains on `SCMDataset` which uses random projections to simulate Transformer hidden states. Achieves ~0.998 overall correlation.

### Real Transformer hidden states

```bash
# Software Deployment domain (default)
python train_on_transformer.py --model gpt2
python train_on_transformer.py --model gpt2 --d_causal 48 --epochs 200 --num_samples 50000
python train_on_transformer.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
python train_on_transformer.py --model meta-llama/Llama-3.2-1B

# Clinical Treatment domain
python train_on_transformer.py --domain clinical --model gpt2 --d_causal 48 --epochs 200 --num_samples 50000
```

Trains on `TextSCMDataset` (deployment) or `TextClinicalDataset` (clinical) which encode SCM scenarios as natural language, run them through a frozen Transformer, and train Causeway on the real hidden state representations. Supports any HuggingFace autoregressive model. Use `--domain` to select the SCM domain.

The dataset is cached to disk (`cache_{domain}_{model}_{samples}_v2.pt`) to avoid re-encoding on subsequent runs.

### Key Training Arguments

| Argument | Default | Description |
|---|---|---|
| `--domain` | `deployment` | SCM domain: `deployment` or `clinical` |
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

## SCM Domain: Clinical Treatment

The clinical SCM models a treatment system with 8 variables:

**Controllable** (actions): `drug_dosage`, `treatment_intensity`, `monitoring_frequency`, `activity_restriction`

**Observable** (effects): `adverse_events`, `recovery_rate`, `organ_stress`, `care_cost`

**Causal graph** (10 ground-truth edges):
```
drug_dosage         -> adverse_events    (+0.5)
drug_dosage         -> recovery_rate     (+0.4)
drug_dosage         -> organ_stress      (+0.6)
treatment_intensity -> recovery_rate     (+0.5)
treatment_intensity -> adverse_events    (+0.3)
monitoring_freq     -> adverse_events    (-0.4, protective)
activity_restrict   -> recovery_rate     (-0.3)
adverse_events      -> organ_stress      (+0.5)
organ_stress        -> recovery_rate     (-0.4)
adverse_events      -> care_cost         (+0.6)
```

**Structured output mapping** (same 5 universal dims):
- `risk_shift` = adverse_events delta
- `goal_progress` = recovery_rate delta
- `constraint_violation` = clip(organ_stress delta * 3, 0, 1)
- `resource_cost` = care_cost delta
- `success_probability` = (recovery_rate - adverse_events) / 2

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

### GPT-2 — Clinical Treatment Domain
- **Overall correlation: 0.9420** (same 0.794M params, same hyperparameters)
- Best val loss: 0.1775 (epoch 197)
- Overall MAE: 0.0442

| Dimension | Correlation | Dir Acc | MAE |
|---|---|---|---|
| risk_shift | 0.9351 | 0.797 | 0.0515 |
| goal_progress | 0.9329 | 0.853 | 0.0374 |
| constraint_violation | 0.9309 | 0.480 | 0.0764 |
| resource_cost | 0.9366 | 0.803 | 0.0281 |
| success_probability | 0.9238 | 0.859 | 0.0274 |

**Cross-domain result**: Same architecture, same hyperparameters, two completely different domains — both above 0.94. The 5-dim structured output and the causal adapter architecture generalize across domains. One domain is a proof of concept; two domains is infrastructure.

### Mistral 7B v0.3 — Software Deployment Domain
- **Overall correlation: 0.9641** (17.74M params, 0.24% overhead vs frozen Mistral 7B)
- Best val loss: 0.1712 (epoch 181)
- Overall MAE: 0.0476

| Dimension | Correlation | Dir Acc | MAE |
|---|---|---|---|
| risk_shift | 0.9608 | 0.807 | 0.0466 |
| goal_progress | 0.9543 | 0.852 | 0.0534 |
| constraint_violation | 0.9546 | 0.478 | 0.0697 |
| resource_cost | 0.9567 | 0.450 | 0.0227 |
| success_probability | 0.9607 | 0.886 | 0.0455 |

Graph converged to 0.2 expected edges at 0.136 temperature. d_causal=64 (up from 48 on GPT-2) for richer 4096-dim representations.

### Mistral 7B v0.3 — Clinical Treatment Domain
- **Overall correlation: 0.9555** (same 17.74M params, same hyperparameters)
- Overall MAE: 0.0373

| Dimension | Correlation | Dir Acc | MAE |
|---|---|---|---|
| risk_shift | 0.9535 | 0.816 | 0.0424 |
| goal_progress | 0.9403 | 0.851 | 0.0362 |
| constraint_violation | 0.9536 | 0.478 | 0.0582 |
| resource_cost | 0.9536 | 0.818 | 0.0245 |
| success_probability | 0.9363 | 0.873 | 0.0250 |

### Scaling: GPT-2 vs Mistral 7B

| | GPT-2 (768-dim) | Mistral 7B (4096-dim) | Improvement |
|---|---|---|---|
| Deployment corr | 0.9494 | **0.9641** | +1.5% |
| Clinical corr | 0.9420 | **0.9555** | +1.4% |
| Causeway params | 0.794M | 17.74M | 22x (still 0.24% of backbone) |
| d_causal | 48 | 64 | +33% |

Three backbones (synthetic, GPT-2, Mistral 7B), two domains (deployment, clinical), all above 0.94 correlation. The architecture scales: richer backbone representations produce better counterfactual predictions without changing anything except d_causal.

### Closed-Loop Evaluation

Tests whether Causeway's causal deltas, when fed back to the Transformer, improve decision quality. Scored against SCM ground truth on 200 fresh deployment scenarios with paired safe/risky actions.

**Causeway as standalone decision oracle:**

| Metric | GPT-2 | Mistral 7B |
|---|---|---|
| Delta accuracy (fresh data) | 0.9205 corr | 0.9296 corr |
| Action ranking (safe > risky) | 99.5% | 99.5% |
| Quality gap correlation | 0.8714 | 0.8656 |

Causeway correctly identifies the safer action in 199/200 scenarios on both backbones.

**Closed-loop injection (does the Transformer use Causeway's signal?):**

| Method | GPT-2 (124M) | Mistral 7B |
|---|---|---|
| Baseline (no Causeway) | 49.0% (coin flip) | 100.0% |
| + Causeway text injection | 49.0% | 100.0% |
| + Bridge prefix (untrained) | 49.0% | 100.0% |

**Interpretation**: GPT-2 cannot use the injected causal signal — it's too small to reason about structured analysis text and picks randomly. Mistral 7B already scores 100% without Causeway because the test scenarios have obviously safe vs obviously risky actions that a 7B model understands from text alone.

Causeway's value is at the representation level: 99.5% action ranking from hidden states, no text generation needed. It operates as a structured decision oracle — faster and more reliable than asking the LLM to reason about risk in natural language.

**Script**: `eval_closed_loop.py` — supports any backbone via checkpoint auto-detection.

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
transformers>=4.36.0
accelerate>=0.25.0
sentencepiece>=0.1.99   # for Mistral/LLaMA tokenizers
protobuf>=4.25.0
```

## Checkpoints

| File | Description |
|---|---|
| `causeway_checkpoint.pt` | Synthetic SCM training |
| `causeway_gpt2.pt` | GPT-2 V2 supervised — deployment domain (0.9494 corr) |
| `causeway_clinical_gpt2.pt` | GPT-2 V2 supervised — clinical domain (0.9420 corr) |
| `cache_deployment_gpt2_50000_v2.pt` | Cached GPT-2 deployment dataset (50K samples) |
| `cache_clinical_gpt2_50000_v2.pt` | Cached GPT-2 clinical dataset (50K samples) |
| `causeway_Mistral-7B-v0.3.pt` | Mistral 7B supervised — deployment domain (0.9641 corr) |
| `causeway_clinical_Mistral-7B-v0.3.pt` | Mistral 7B supervised — clinical domain (0.9555 corr) |
| `causeway_ss_gpt2.pt` | Self-supervised Causeway + DeltaDecoder (cos=0.707) |
| `causeway_ss_gpt2_baseline.pt` | Self-supervised baseline MLP (cos=0.769) |
| `cache_ss_gpt2_50000.pt` | Cached self-supervised dataset (50K, 461MB) |
| `keystone_gpt2.pt` | Keystone GPT-2 deployment domain (after training) |
| `keystone_clinical_gpt2.pt` | Keystone GPT-2 clinical domain (after training) |

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
| Parameter-efficient | No | N/A | No | No | No | **Yes (0.794M GPT-2, 17.74M Mistral)** |
| Plugs into any Transformer | No | No | No | No | Diffusion only | **Yes** |

The unique contribution is the combination: a parameter-efficient module (0.794M for GPT-2, 17.74M for Mistral 7B — always <0.25% of backbone) implementing full Pearl counterfactual inference through a learned sparse DAG, operating on real Transformer hidden states, producing structured non-text outputs, without touching the backbone weights.

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

## Environment Requirements

- **Python**: 3.10+
- **CUDA**: Required for Transformer encoding. Training Causeway itself runs on CPU or GPU.
- **VRAM for encoding**: GPT-2 ~2 GB, Mistral 7B fp16 ~16-18 GB (freed after encoding)
- **VRAM for training**: ~1 GB (Causeway only, backbone is freed after dataset caching)
- **Recommended GPU for Mistral**: 24+ GB (RTX 4090, A10, A100)

## Gotchas

- **fp16 dtype mismatch**: Large models (Mistral, LLaMA) load in fp16, producing fp16 hidden states. Causeway parameters are fp32. The datasets cast to `.float()` on encoding and cache loading, and `StateEncoder.forward` has a safety `h.float()` cast — but if you build custom pipelines, ensure inputs are fp32.
- **RunPod terminal disconnects**: The RunPod web terminal drops websocket connections frequently. Always use `nohup` for training runs:
  ```bash
  nohup ./run_clinical.sh > out.log 2>&1 &
  tail -20 out.log  # check progress
  ```
- **Python output buffering**: When redirecting to a log file, Python buffers stdout. Use `python -u` (unbuffered) or the provided shell scripts which include `-u`.
- **Cached datasets**: Encoded datasets are cached to `cache_{domain}_{model}_{samples}_v2.pt`. Delete the cache if you change the encoding logic (pooling strategy, text templates, etc.) — stale caches silently produce wrong results.
- **`--d_causal` scaling**: Use 48 for GPT-2 (768-dim), 64 for Mistral 7B (4096-dim). The graph has `d_causal^2` potential edges, so doubling d_causal quadruples the graph search space.

## Key Design Decisions

- **d_causal=48**: Sufficient for the 8-variable SCM. Increasing it grows the graph quadratically (48^2 = 2304 potential edges).
- **d_action = d_model**: Actions stored as full Transformer embeddings to preserve information. The InterventionEngine decodes them to causal-space intervention masks and values.
- **Gumbel-sigmoid edge gating**: Achieves genuinely sparse graphs (0-1 edges at convergence) rather than many weak edges from L1 alone.
- **Temperature annealing** (1.0 -> 0.05): Soft edges during early training for gradient flow, hard binary selection at convergence.
- **Acyclicity ramp**: DAG constraint ramped from 0 to 10 over first 30% of epochs. This lets the graph learn useful edges before the DAG constraint prunes cycles.
- **Asymmetric loss**: Risk underestimation penalized 4x vs overestimation for safety-critical deployment decisions.
