# 🏷️ Amazon Price Predictor Fine Tuning LLM and Deep-Neural Network-Pipeline

## 📊 Model Performance Comparison

![Model Performance](graph/graph%20model%20performance%20comparator%20for%20pricer%20final.jpg)

## 📈 Fine-Tuning Training (Weights & Biases)

![WandB Training](graph/wandb%20finetuning%20training%20screenshot.png)

LLaMA 3.2 fine-tuné sur le dataset complet ($39.85) bat GPT-5.1 ($44.74), Claude 4.5 Sonnet ($47.10) et tous les modèles frontier testés — sans accès aux poids propriétaires, uniquement grâce à la spécialisation sur les données Amazon.

> **English version below** / Version française ci-dessus

---

## 🇫🇷 Version Française

### Vue d'ensemble

> *Hybrid ML pipeline combining LLM-based structured summarization with deep neural network regression and LLaMA 3.2 fine-tuning for price prediction — outperforming frontier models on this specific task.*

Ce projet implémente un pipeline ML de bout en bout pour prédire le prix d'un produit Amazon à partir de sa description textuelle. L'idée centrale : utiliser un LLM pour structurer et nettoyer la donnée brute, puis tester plusieurs approches en montant progressivement en puissance — jusqu'au fine-tuning de **LLaMA 3.2** sur **+820 000 fiches produits**, avec des résultats qui **surpassent les meilleurs modèles de la planète** sur cette tâche spécifique.

**Points forts du pipeline :**
- 🧹 **Data Cleaning** — nettoyage rigoureux sur **+820 000 produits** : filtrage des prix, suppression des SKU, part numbers et champs parasites
- 🤖 **LLM Preprocessing** — transformation des descriptions brutes en résumés structurés via Groq, en **batches de 1 000 items** asynchrones
- 🧠 **Deep Neural Network** — réseau ResNet-style de **10 couches**, **4 096 neurones** par couche, **+100 millions de paramètres**
- 🦙 **Fine-tuning LLaMA 3.2** — résultats supérieurs à GPT-5.1 et Claude Opus sur cette tâche de pricing
- 📊 **Évaluation statistique** — MAE, MSE, R², courbes d'erreur avec intervalles de confiance à 95 %
- ⚡ **Scalabilité** — traitement multi-processus (ProcessPool), batch asynchrone JSONL, support CUDA/MPS/CPU

---

### 🗺️ Progression du projet

Ce projet explore une montée en puissance progressive de 6 approches :

| Étape | Approche | Résultats |
|-------|----------|-----------|
| 1️⃣ **Data Curation** | Chargement et nettoyage de +820 000 produits Amazon | — |
| 2️⃣ **Data Pre-processing** | Résumés LLM via Groq — batches de 1 000 items | — |
| 3️⃣ **Baselines & ML classique** | Random Forest, XGBoost | Référence |
| 4️⃣ **Deep Neural Network** | DNN ResNet-style : 10 couches, 4 096 neurones | ✅ OK |
| 5️⃣ **Fine-tuning Frontier** | Fine-tuning GPT sur +820 000 exemples | ⚠️ Moyen |
| 6️⃣ **Fine-tuning LLaMA 3.2** | Fine-tuning open-source sur dataset complet | 🏆 SOTA |

> 🏆 Le fine-tuning de **LLaMA 3.2** sur ce dataset spécifique surpasse **GPT-5, Claude Opus et les meilleurs modèles frontier** sur la tâche de prédiction de prix Amazon.

---

### 📊 Dataset

Source : **McAuley-Lab/Amazon-Reviews-2023** (HuggingFace)

| Version | Taille | Usage |
|---------|--------|-------|
| 🪶 Lite | ~22 000 produits | Développement & tests rapides |
| 🔥 Full | ~820 000 produits | Entraînement complet |

Le dataset est nettoyé, enrichi par LLM, puis poussé sur le **HuggingFace Hub** pour être réutilisé à chaque étape du pipeline.

---

### ⚙️ Installation

```bash
git clone https://github.com/ton-utilisateur/amazon-price-prediction-llm-deep-neural-network.git
cd amazon-price-prediction-llm-deep-neural-network
pip install -r requirements.txt
```

**Variables d'environnement** — crée un fichier `.env` :

```env
GROQ_API_KEY=ta_clé_groq_ici
```

---

### 📦 Dépendances principales

| Librairie | Usage |
|-----------|-------|
| `torch` | Réseau de neurones (PyTorch) |
| `transformers` | Fine-tuning LLaMA 3.2 |
| `pydantic` | Validation du modèle de données |
| `datasets` | Chargement du dataset HuggingFace |
| `scikit-learn` | Vectorisation & métriques |
| `groq` / `litellm` | Appels LLM (résumés) |
| `plotly` | Visualisations interactives |
| `tqdm` | Barres de progression |
| `python-dotenv` | Gestion des variables d'environnement |

---

### 🗂️ Description des modules

#### `pricer/items.py` — Modèle de données
Définit la classe `Item` via **Pydantic**. Chaque item représente un produit Amazon avec ses champs : `title`, `category`, `price`, `full` (texte brut jusqu'à **4 000 caractères**), `summary` (résumé LLM), `prompt`, etc.

Fonctionnalités clés :
- `make_prompt()` — génère le prompt d'entraînement
- `test_prompt()` — retourne le prompt sans le prix (pour inférence)
- `push_to_hub()` / `from_hub()` — intégration HuggingFace Hub

#### `pricer/loaders.py` — Chargement du dataset
Charge le dataset **McAuley-Lab/Amazon-Reviews-2023** depuis HuggingFace (**+820 000 produits** dans la version complète). Face à cette volumétrie, un simple chargement séquentiel serait prohibitif : le module utilise un `ProcessPoolExecutor` pour distribuer le travail sur tous les cœurs CPU disponibles, en découpant le dataset en **chunks de 1 000 éléments** traités en parallèle.

#### `pricer/parser.py` — Nettoyage des données
Cœur du data engineering. Filtre et nettoie chaque produit :
- Plage de prix acceptée : **$0.50 → $999.49**
- Supprime les numéros de pièces, SKU alphanumériques et champs parasites (Best Sellers Rank, numéros de modèle, etc.)
- Normalise les poids en livres (supporte : pounds, ounces, grams, milligrams, kilograms)
- Limite le texte à **4 000 caractères** max (3 000 par champ)
- Exige un minimum de **600 caractères** de contenu — les fiches trop courtes sont exclues

> ⚠️ Un mauvais nettoyage = un modèle qui apprend du bruit. Cette étape est critique.

#### `pricer/preprocessor.py` — Résumé LLM (unitaire)
Contient la classe `Preprocessor` qui utilise **litellm** pour appeler le modèle `groq/openai/gpt-oss-20b` et transformer une description produit longue en résumé structuré en **5 champs** :

```
Title / Category / Brand / Description / Details
```

Suit également les **tokens consommés et le coût total** des appels API (`total_input_tokens`, `total_output_tokens`, `total_cost`) — indispensable pour monitorer les dépenses à grande échelle.

#### `pricer/batch.py` — Résumé LLM (batch scalable)
Version scalable du preprocessor. Traite le dataset entier en **batches de 1 000 items** via des jobs asynchrones Groq :
1. Génère des fichiers `.jsonl` par batch de **1 000 items**
2. Upload les fichiers sur Groq
3. Lance des jobs asynchrones (fenêtre de **24h**)
4. Récupère et applique les résultats
5. Sauvegarde l'état en `.pkl` pour reprendre si interruption

> 💡 Sur **820 000 items**, cela représente **820 batches** traités de façon asynchrone — beaucoup plus économique que de payer 820 000 appels API individuels, et sans risque de timeout.

#### `pricer/deepneuralnetwork.py` — Modèle DNN

Un **DNN (Deep Neural Network)** désigne un réseau de neurones comportant de nombreuses couches cachées — par opposition à un réseau superficiel. Ici, l'architecture pousse la profondeur à l'extrême pour capturer des relations complexes entre les mots d'une description et le prix d'un produit :

| Paramètre | Valeur |
|-----------|--------|
| Couches cachées | **10** |
| Neurones par couche | **4 096** |
| Features d'entrée | **5 000** (HashingVectorizer) |
| Paramètres entraînables | **~100 millions+** |
| Blocs résiduels | **8** (skip connections ResNet-style) |
| Batch size | **64** |
| Optimiseur | AdamW (lr=0.001, weight_decay=0.01) |
| Scheduler | CosineAnnealingLR |

```
Input (5 000) → Linear(4 096) → [ResidualBlock x8] → Linear(1) → Prix prédit
```

Les **blocs résiduels** (skip connections inspirées de ResNet) sont la clé pour entraîner un réseau aussi profond sans que le gradient ne disparaisse. Les prix sont **transformés en log-scale** puis standardisés avant l'entraînement, ce qui stabilise considérablement la convergence. Le device est détecté automatiquement au lancement : **CUDA > MPS (Apple Silicon) > CPU**.

#### `pricer/evaluator.py` — Évaluation
Évalue n'importe quelle fonction de prédiction sur un échantillon (par défaut **200 points**) en parallèle via `ThreadPoolExecutor` (**5 workers**) :
- **Métriques** : MAE (erreur absolue moyenne en $), MSE, R²
- **Scatter plot** : prix prédit vs prix réel, coloré par précision (🟢 erreur < $40 ou < 20%, 🟡 < $80 ou < 40%, 🔴 au-delà)
- **Courbe d'erreur** : erreur cumulée avec intervalle de confiance à **95 %**

---

#### `frontier/` — Fine-tuning modèles frontier (GPT)

Première tentative de fine-tuning sur un modèle propriétaire. Les résultats se sont révélés **moyens** : malgré l'utilisation de **+820 000 exemples**, le modèle peinait à capturer la complexité des prix Amazon, probablement en raison des contraintes inhérentes aux modèles fermés (accès limité aux poids, coût élevé, customisation restreinte).

- `items.py` — version adaptée du modèle de données pour le format fine-tuning
- `evaluator.py` — évaluation comparative des résultats frontier
- `utils.py` — utilitaires d'entraînement et de gestion des données

---

#### `llama3.2/` — Fine-tuning LLaMA 3.2 🏆

L'approche finale et la plus performante. En fine-tunant **LLaMA 3.2** (modèle open-source) sur le dataset complet de **+820 000 fiches produits Amazon**, les résultats dépassent ceux obtenus avec les modèles frontier propriétaires — et surpassent **GPT-4o, Claude et les meilleurs modèles disponibles** sur cette tâche spécifique de prédiction de prix.

Pourquoi LLaMA 3.2 surpasse les modèles frontier ici :
- **Spécialisation totale** — accès complet aux poids pour un fine-tuning profond
- **Volume de données** — +820 000 exemples domaine-spécifique vs connaissance généraliste
- **Format adapté** — format `prompt/completion` optimisé pour la tâche

`items.py` — version étendue avec support tokenizer, gestion de la longueur de séquence et format `prompt/completion` :
- `make_prompts()` — génère prompt + completion en tronquant intelligemment au max de tokens
- `count_tokens()` / `count_prompt_tokens()` — contrôle précis de la longueur des séquences
- `push_prompts_to_hub()` — pousse le dataset en format SFT (Supervised Fine-Tuning) sur HuggingFace

- `evaluator.py` — évaluation avec parsing robuste des sorties LLM et intervalles de confiance à 95 %
- `utils.py` — utilitaires d'entraînement

---

### 🚀 Utilisation

```python
# 1. Charger les données (~820 000 produits)
from pricer.loaders import ItemLoader
items = ItemLoader("Electronics").load()

# 2. Générer les résumés en batch (820 batches de 1 000 items)
from pricer.batch import Batch
Batch.create(items, lite=False)
Batch.run()
Batch.fetch()

# 3. Entraîner le DNN (10 couches, 4 096 neurones, ~100M paramètres)
from pricer.deepneuralnetwork import DeepNeuralNetworkRunner
runner = DeepNeuralNetworkRunner(train_items, val_items)
runner.setup()
runner.train(epochs=10)
runner.save("model.pt")

# 4. Fine-tuner LLaMA 3.2 (résultats SOTA)
# Voir llama3.2/ pour les scripts d'entraînement

# 5. Évaluer sur 200 points
from pricer.evaluator import evaluate
evaluate(runner.inference, test_items, size=200)
```

---

### 📊 Métriques d'évaluation

| Métrique | Description |
|----------|-------------|
| **MAE** | Erreur absolue moyenne en dollars |
| **MSE** | Erreur quadratique moyenne |
| **R²** | Coefficient de détermination (% variance expliquée) |

---

---

## 🇬🇧 English Version

### Overview

> *Hybrid ML pipeline combining LLM-based structured summarization with deep neural network regression and LLaMA 3.2 fine-tuning for price prediction — outperforming frontier models on this specific task.*

This project builds an end-to-end ML pipeline to predict an Amazon product's price from its text description. The core idea: use an LLM to structure and clean raw product data, then test progressively more powerful approaches — culminating in the fine-tuning of **LLaMA 3.2** on **820,000+ product listings**, with results that **outperform the best frontier models** on this specific task.

**Pipeline highlights:**
- 🧹 **Data Cleaning** — rigorous cleaning across **820,000+ products**: price filtering, SKU removal, part numbers and junk field stripping
- 🤖 **LLM Preprocessing** — transforms raw descriptions into structured summaries via Groq, in async **batches of 1,000 items**
- 🧠 **Deep Neural Network** — ResNet-style network with **10 layers**, **4,096 neurons** per layer, **100M+ parameters**
- 🦙 **LLaMA 3.2 Fine-tuning** — results outperforming GPT-4o and Claude on this pricing task
- 📊 **Statistical Evaluation** — MAE, MSE, R², error curves with 95% confidence intervals
- ⚡ **Scalability** — multi-process loading (ProcessPool), async JSONL batch jobs, CUDA/MPS/CPU auto-detection

---

### 🗺️ Project Progression

This project explores a progressive scale-up across 6 approaches:

| Step | Approach | Results |
|------|----------|---------|
| 1️⃣ **Data Curation** | Loading and cleaning 820,000+ Amazon products | — |
| 2️⃣ **Data Pre-processing** | LLM summaries via Groq — batches of 1,000 items | — |
| 3️⃣ **Baselines & Classic ML** | Random Forest, XGBoost | Baseline |
| 4️⃣ **Deep Neural Network** | ResNet-style DNN: 10 layers, 4,096 neurons | ✅ OK |
| 5️⃣ **Frontier Fine-tuning** | GPT fine-tuning on 820,000+ examples | ⚠️ Average |
| 6️⃣ **LLaMA 3.2 Fine-tuning** | Open-source fine-tuning on full dataset | 🏆 SOTA |

> 🏆 Fine-tuning **LLaMA 3.2** on this domain-specific dataset outperforms **GPT-5.1, Claude Sonnet 4.5 and the best available frontier models** on the Amazon price prediction task.

---

### 📊 Dataset

Source: **McAuley-Lab/Amazon-Reviews-2023** (HuggingFace)

| Version | Size | Usage |
|---------|------|-------|
| 🪶 Lite | ~22,000 products | Development & fast iteration |
| 🔥 Full | ~820,000 products | Full training |

The dataset is cleaned, LLM-enriched, then pushed to the **HuggingFace Hub** and reused at each stage of the pipeline.

---

### 🗃️ Project Structure

```
amazon-price-prediction-llm-deep-neural-network/
├── pricer/                        → Core pipeline (data + DNN)
│   ├── items.py                   → Pydantic data model
│   ├── parser.py                  → Raw data cleaning and filtering
│   ├── loaders.py                 → Parallel loading of 820,000+ products
│   ├── preprocessor.py            → LLM summarization (single call)
│   ├── batch.py                   → Bulk LLM summarization (1,000 items/batch)
│   ├── deepneuralnetwork.py       → DNN: 10 layers, 4,096 neurons, residual blocks
│   └── evaluator.py               → MAE/MSE/R² evaluation with visualizations
│
├── frontier/                      → Frontier model fine-tuning (GPT)
│   ├── items.py                   → Data model adapted for fine-tuning format
│   ├── evaluator.py               → Frontier model evaluation
│   └── utils.py                   → Utilities
│
└── llama3.2/                      → LLaMA 3.2 fine-tuning 🏆
    ├── items.py                   → Extended model (prompt/completion + tokenizer)
    ├── evaluator.py               → Evaluation with 95% confidence intervals
    └── utils.py                   → Training utilities
```

---

### ⚙️ Installation

```bash
git clone https://github.com/your-username/amazon-price-prediction-llm-deep-neural-network.git
cd amazon-price-prediction-llm-deep-neural-network
pip install -r requirements.txt
```

**Environment variables** — create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key_here
```

---

### 📦 Main Dependencies

| Library | Usage |
|---------|-------|
| `torch` | Neural network (PyTorch) |
| `transformers` | LLaMA 3.2 fine-tuning |
| `pydantic` | Data model validation |
| `datasets` | HuggingFace dataset loading |
| `scikit-learn` | Vectorization & metrics |
| `groq` / `litellm` | LLM API calls (summaries) |
| `plotly` | Interactive visualizations |
| `tqdm` | Progress bars |
| `python-dotenv` | Environment variable management |

---

### 🗂️ Module Descriptions

#### `pricer/items.py` — Data Model
Defines the `Item` class using **Pydantic**. Each item represents an Amazon product with fields: `title`, `category`, `price`, `full` (raw text up to **4,000 characters**), `summary` (LLM summary), `prompt`, etc.

Key methods:
- `make_prompt()` — generates the training prompt
- `test_prompt()` — returns the prompt without the price (for inference)
- `push_to_hub()` / `from_hub()` — HuggingFace Hub integration

#### `pricer/loaders.py` — Dataset Loading
Loads the **McAuley-Lab/Amazon-Reviews-2023** dataset from HuggingFace (**820,000+ products** in full mode). At this scale, sequential loading would be prohibitive: the module uses a `ProcessPoolExecutor` to distribute work across all available CPU cores, splitting the dataset into **chunks of 1,000 items** processed in parallel.

#### `pricer/parser.py` — Data Cleaning
The data engineering core. Filters and cleans each product:
- Accepted price range: **$0.50 → $999.49**
- Removes part numbers, alphanumeric SKUs and noisy fields (Best Sellers Rank, model numbers, etc.)
- Normalizes weights to pounds (supports: pounds, ounces, grams, milligrams, kilograms)
- Caps text at **4,000 characters** max (3,000 per field)
- Requires a minimum of **600 characters** of content — short listings are excluded

> ⚠️ Poor cleaning means the model learns noise instead of signal. This step is critical.

#### `pricer/preprocessor.py` — LLM Summarization (single call)
Contains the `Preprocessor` class that uses **litellm** to call the `groq/openai/gpt-oss-20b` model and transform a long product description into a structured **5-field summary**:

```
Title / Category / Brand / Description / Details
```

Also tracks **token usage and total API cost** (`total_input_tokens`, `total_output_tokens`, `total_cost`) — essential for monitoring spending at scale.

#### `pricer/batch.py` — LLM Summarization (scalable batch)
Scalable version of the preprocessor. Processes the entire dataset in **batches of 1,000 items** via async Groq jobs:
1. Generates `.jsonl` files in batches of **1,000 items**
2. Uploads files to Groq
3. Launches async jobs (completion window: **24h**)
4. Fetches and applies results
5. Saves state as `.pkl` to resume after interruption

> 💡 On **820,000 items**, this means **820 batches** processed asynchronously — far more cost-effective than paying for 820,000 individual API calls, and with no timeout risk.

#### `pricer/deepneuralnetwork.py` — DNN Model

A **DNN (Deep Neural Network)** is a neural network with many hidden layers — as opposed to shallow architectures. Here, the depth is pushed to the extreme to capture complex relationships between product description words and price:

| Parameter | Value |
|-----------|-------|
| Hidden layers | **10** |
| Neurons per layer | **4,096** |
| Input features | **5,000** (HashingVectorizer) |
| Trainable parameters | **~100M+** |
| Residual blocks | **8** (ResNet-style skip connections) |
| Batch size | **64** |
| Optimizer | AdamW (lr=0.001, weight_decay=0.01) |
| Scheduler | CosineAnnealingLR |

```
Input (5,000) → Linear(4,096) → [ResidualBlock x8] → Linear(1) → Predicted price
```

**Residual blocks** (ResNet-inspired skip connections) are the key to training a network this deep without gradients vanishing. Prices are **log-transformed** then standardized before training, which significantly stabilizes convergence. The device is auto-detected at startup: **CUDA > MPS (Apple Silicon) > CPU**.

#### `pricer/evaluator.py` — Evaluation
Evaluates any prediction function on a data sample (default **200 points**) in parallel via `ThreadPoolExecutor` (**5 workers**):
- **Metrics**: MAE (mean absolute error in $), MSE, R²
- **Scatter plot**: predicted vs actual price, color-coded by accuracy (🟢 error < $40 or < 20%, 🟡 < $80 or < 40%, 🔴 beyond)
- **Error curve**: cumulative error with **95% confidence interval**

---

#### `frontier/` — Frontier Model Fine-tuning (GPT)

The first fine-tuning attempt, using a proprietary frontier model. Despite training on **820,000+ examples**, results were **average**: the model struggled to match the complexity of Amazon pricing, likely due to the inherent constraints of closed models (limited weight access, high cost, restricted customization).

- `items.py` — adapted data model for the fine-tuning format
- `evaluator.py` — comparative evaluation of frontier results
- `utils.py` — training and data utilities

---

#### `llama3.2/` — LLaMA 3.2 Fine-tuning 🏆

The final and most powerful approach. By fine-tuning **LLaMA 3.2** (open-source) on the full dataset of **820,000+ Amazon product listings**, the results exceed those of proprietary frontier models — outperforming **GPT-5.1, Claude Sonnet and the best available models** on this specific pricing task.

Why LLaMA 3.2 outperforms frontier models here:
- **Total specialization** — full weight access enables deep fine-tuning
- **Data volume** — 820,000+ domain-specific examples vs generalist knowledge
- **Optimized format** — `prompt/completion` format tailored to the task

`items.py` — extended version with tokenizer support, sequence length management and `prompt/completion` format:
- `make_prompts()` — generates prompt + completion with intelligent token truncation
- `count_tokens()` / `count_prompt_tokens()` — precise sequence length control
- `push_prompts_to_hub()` — pushes dataset in SFT (Supervised Fine-Tuning) format to HuggingFace

- `evaluator.py` — evaluation with robust LLM output parsing and 95% confidence intervals
- `utils.py` — training utilities

---

### 🚀 Usage

```python
# 1. Load data (~820,000 products)
from pricer.loaders import ItemLoader
items = ItemLoader("Electronics").load()

# 2. Generate summaries in batch (820 batches of 1,000 items)
from pricer.batch import Batch
Batch.create(items, lite=False)
Batch.run()
Batch.fetch()

# 3. Train the DNN (10 layers, 4,096 neurons, ~100M parameters)
from pricer.deepneuralnetwork import DeepNeuralNetworkRunner
runner = DeepNeuralNetworkRunner(train_items, val_items)
runner.setup()
runner.train(epochs=10)
runner.save("model.pt")

# 4. Fine-tune LLaMA 3.2 (SOTA results)
# See llama3.2/ for training scripts

# 5. Evaluate on 200 data points
from pricer.evaluator import evaluate
evaluate(runner.inference, test_items, size=200)
```

---

### 📊 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **MAE** | Mean Absolute Error in dollars |
| **MSE** | Mean Squared Error |
| **R²** | Coefficient of determination (% variance explained) |

---

### 📄 License

MIT License — feel free to use, modify, and distribute.
