# Prototype-Guided Backdoor Defense for Malware Classifiers

This project implements a post-training backdoor defense for malware classification models. The main goal is to detect and remove malicious behavior added into the model through backdoor attacks.

In a backdoor attack, attackers add hidden triggers into the training data so that the model works normally on clean inputs, but gives wrong predictions whenever the trigger appears. Our defense works after training, so we do not need the original training data again. We only use a small clean dataset to purify the infected model.


# What We Did

The main idea of the defense is to find suspicious neurons that may have learned the backdoor trigger, remove them, and then fine-tune the model to recover clean accuracy.

## Dynamic Thresholding using MAD

Earlier, we were using a fixed threshold after Min-Max normalization to detect suspicious neurons. The problem was that Min-Max scaling always makes the largest score equal to 1. Because of this, even normal neurons could cross the threshold and get removed unnecessarily. Also, if one neuron had a very large score, smaller malicious neurons could become too small and remain undetected.

To solve this, we switched to Median Absolute Deviation (MAD) based dynamic thresholding. Instead of using one fixed threshold, the threshold is calculated separately for each layer based on its neuron scores.

First, the median score is calculated, then MAD is computed as:

```text
MAD = median(|xi - median(x)|)
```

The threshold becomes:

```text
Threshold = median + k × MAD
```

where `k` is usually `3`.

Neurons whose scores are much higher than the normal range are treated as suspicious and removed.

This approach is more stable, reduces false positives, preserves clean accuracy better, and works well for different layers and architectures.


# How It Works (Project Workflow)

Here is a short breakdown of how the project operates using the core files:

- **`src/backdoor.py` (Attack Phase)**: Simulates the attacker by injecting a hidden trigger into clean malware samples and changing their labels to "Benign" to create poisoned data.

- **`train_poisoned_model.py` (Training Phase)**: Trains the neural network (`MalwareMLP`) on this poisoned dataset. The model learns to classify clean data correctly but fails on triggered data.

- **`src/purification.py` (Defense Phase)**: Contains the core defense logic. It analyzes neuron activations on clean data, applies Dynamic MAD Thresholding to find suspicious backdoor neurons, removes them, and fine-tunes the model.

- **`final_evaluation.py` (Evaluation Phase)**: Tests both the poisoned and purified models to verify that the defense removes the backdoor while preserving clean accuracy.


# Instructions to Run

## Install Requirements

```bash
pip install -r requirements.txt
```

## Train the Poisoned Model and Run Purification

```bash
python train_poisoned_model.py --max_epochs 5
```

This command:
- Creates poisoned data
- Trains the backdoored malware classifier
- Runs the purification process
- Saves the purified model


## Run Final Evaluation

```bash
python final_evaluation.py
```

This script:
- Evaluates the poisoned model
- Evaluates the purified model
- Calculates metrics like Clean Accuracy (CA), Attack Success Rate (ASR), Precision, Recall, and F1-Score
- Generates result graphs and confusion matrices


# Expected Output

After purification:
- Attack Success Rate (ASR) should reduce close to `0%`
- Clean Accuracy (CA) should remain reasonably high
- Result graphs and evaluation outputs will be saved inside the `outputs/graphs/` folder
