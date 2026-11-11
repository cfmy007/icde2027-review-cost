# Generating review rankings

The files in `main/inputs` contain the original text, the selected claim and reference label, the fixed fit, source and target assignments, and the target tie order. The 13 text collections use the same TF-IDF, Qwen representation, classifiers and Cleanlab ranking as the cross-corpus experiment. `AudioSet.jsonl.gz` contains the corresponding fixed assignments for the official VGGish features and the weak and strong annotation releases. The fit labels and reference labels are the selected labels from the study; collecting these labels is outside the record-level review budget.

Run the 13 text collections and AudioSet with Python 3.12, a CUDA GPU, NumPy, SciPy, scikit-learn, PyTorch, transformers and Cleanlab 2.7.1:

```sh
python main/generate.py AudioSet GoEmotions Wikipedia-Attack D3code Measuring-Hate-Speech Unhealthy-Comments Stanford-Politeness SBIC MBIC DynaSent SNLI MultiNLI HateXplain Davidson --model /path/to/Qwen2.5-0.5B-Instruct
python main/compare.py
```

The model is available at https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct. The study used Qwen2.5-0.5B-Instruct, Python 3.12.3, NumPy 2.3.5, SciPy 1.18.1, scikit-learn 1.9.1, PyTorch 2.13.0 with CUDA 13, transformers 5.15.0, and Cleanlab 2.7.1. The archived run used NVIDIA RTX 3090 GPUs.

To reconstruct the AudioSet inputs from the included weak annotations, strong annotations, ontology and released VGGish features:

```sh
mkdir -p /tmp/audioset-study
tar -xzf main/audioset-inputs.tar.gz -C /tmp/audioset-study
python main/prepare_audio.py /tmp/audioset-study
python main/generate.py AudioSet
```

The included annotation and VGGish feature records cover the 12,334 clips used by this experiment. The fixed input file supplies the clip list and split assignments. Sources: https://research.google.com/audioset/download.html and https://research.google.com/audioset/download_strong.html.

The `authorship` target uses the Human Detectors dataset by Russell et al. ([repository](https://github.com/jenna-russell/human_detectors), [paper](https://aclanthology.org/2025.acl-long.267/)). The source repository is MIT-licensed; its notice is included in `text/HUMAN_DETECTORS_LICENSE.txt`.

The `text/source` files contain the staged text and human judgments for the 23 independent crowd targets. `text/inputs` fixes the selected reference, claim, group, split and tie order. Run their text-to-ranking path with:

```sh
python text/run_all.py
```

After generating both sets of rankings, run `python raw_rankings/replay_generated.py` from the artifact root to recompute the paper's tables in a temporary copy.

This step was run with NumPy 2.2.6, SciPy 1.15.2 and scikit-learn 1.6.1. Each target's generated ordering, reference disagreements, group memberships and source choices were compared with the archived records. The upstream staging selected the votes and references from the original releases before this package; `text/source` retains the staged votes. Public release links and reference definitions are in the artifact README and the paper.

The 14 main targets were rerun in the original GPU environment. All source choices, labels, group memberships and fixed-budget hit counts matched the archive. In MultiNLI, eight positions in three rankings exchanged places, all below the 20% review prefix. The other 13 main rankings matched position by position. The saved paper results remain unchanged.

The complete public releases are available from [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions), [Wikipedia Detox](https://meta.wikimedia.org/wiki/Research:Detox/Data_Release), [D3code](https://aclanthology.org/2024.emnlp-main.1029/), [Measuring Hate Speech](https://aclanthology.org/2022.nlperspectives-1.11/), [Unhealthy Comments](https://aclanthology.org/2020.alw-1.15/), [Stanford Politeness](https://arxiv.org/abs/1306.6078), [SBIC](https://maartensap.com/social-bias-frames/), [MBIC](https://github.com/Media-Bias-Group/MBIC-A-Media-Bias-Annotation-Dataset-Including-Annotator-Characteristics), [DynaSent](https://github.com/cgpotts/dynasent), [SNLI](https://nlp.stanford.edu/projects/snli/), [MultiNLI](https://cims.nyu.edu/~sbowman/multinli/), [HateXplain](https://github.com/hate-alert/HateXplain) and [Davidson](https://github.com/t-davidson/hate-speech-and-offensive-language). Sources for the 23 crowd targets are recorded in `text/sources.json`.
