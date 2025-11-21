import os
import json
import random
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

from datasets import Dataset
from datasets.dataset_dict import DatasetDict

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments
)

from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve
)
import scipy.special

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train and evaluate models with custom config options')
    
    parser.add_argument('--config', type=str, default='config.json', help='Path to the config file')
    
    parser.add_argument('--base', type=str, help='Base name for output directory')
    parser.add_argument('--train_path', type=str, help='Path to training data')
    parser.add_argument('--val_path', type=str, help='Path to validation data')
    parser.add_argument('--model_type', type=str, help='Model type (e.g., bert-base-uncased)')
    parser.add_argument('--num_classes', type=int, help='Number of output classes')
    parser.add_argument('--freeze_base_layers', action='store_true', help='Freeze base layers of the model')
    
    args, unknown = parser.parse_known_args()
    
    training_args_dict = {}
    i = 0
    while i < len(unknown):
        arg = unknown[i]
        if arg.startswith('--training.'):
            param_name = arg[11:]
            if i + 1 < len(unknown) and not unknown[i + 1].startswith('--'):
                training_args_dict[param_name] = unknown[i + 1]
                i += 2
            else:
                training_args_dict[param_name] = True
                i += 1
        else:
            i += 1
    
    args.training_args_dict = training_args_dict
    
    return args

def load_config(config_path="config.json"):
    with open(config_path, 'r') as f:
        return json.load(f)

def update_config_with_args(config, args):
    args_dict = {k: v for k, v in vars(args).items() 
                if v is not None and k not in ['config', 'training_args_dict']}
    
    for param in ['base', 'train_path', 'val_path', 'model_type', 'num_classes', 'freeze_base_layers']:
        if param in args_dict:
            config[param] = args_dict[param]
    
    if hasattr(args, 'training_args_dict') and args.training_args_dict:
        if 'training' not in config:
            config['training'] = {}
            
        for k, v in args.training_args_dict.items():
            config['training'][k] = v
    
    return config

def set_seeds(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def oversample_minority_class(pos, neg, target_ratio=1.0):
    multiplier = int(len(neg) * target_ratio / len(pos))
    oversampled_pos = pos * multiplier
    return oversampled_pos, neg

def create_dict(df, target_column='ETH_BOARD', seed=123, batch_size=16, gradient_accumulation_steps=1):
    random.seed(seed)

    pos = df[df[target_column] == 'yes']
    posPhrases = list(pos['SENTENCE'])

    train_pos = random.sample(range(len(pos)), int(len(pos)*0.67))
    test_pos = [i for i in range(len(pos)) if i not in train_pos]
    train_posPhrases = [posPhrases[i] for i in train_pos]

    neg = df[df[target_column] == 'no']
    negPhrases = list(neg['SENTENCE'])

    train_neg = random.sample(range(len(neg)), int(len(neg)*0.67))
    test_neg = [n for n in range(len(neg)) if n not in train_neg]
    train_negPhrases = [negPhrases[n] for n in train_neg]

    train_os_posPhrases, train_os_negPhrases = oversample_minority_class(
        train_posPhrases, train_negPhrases, target_ratio=1.0
    )
    
    pos_count = len(train_os_posPhrases)
    neg_count = len(train_os_negPhrases)
    total_examples = pos_count + neg_count
    
    effective_batch_size = batch_size * gradient_accumulation_steps
    num_batches = total_examples // effective_batch_size
    
    chunk_multiplier = 2  
    chunks = max(10, num_batches * chunk_multiplier)  
    
    train_x = []
    train_y = []
    
    pos_per_chunk = pos_count // chunks
    neg_per_chunk = neg_count // chunks
    
    pos_samples = list(zip(train_os_posPhrases, [1] * pos_count))
    neg_samples = list(zip(train_os_negPhrases, [0] * neg_count))
    random.shuffle(pos_samples)
    random.shuffle(neg_samples)
    
    pos_idx = 0
    neg_idx = 0
    
    for i in range(chunks):
        chunk_pos = pos_samples[pos_idx:pos_idx + pos_per_chunk]
        chunk_neg = neg_samples[neg_idx:neg_idx + neg_per_chunk]
        
        if i == chunks - 1:
            chunk_pos = pos_samples[pos_idx:]
            chunk_neg = neg_samples[neg_idx:]
        
        chunk = chunk_pos + chunk_neg
        random.shuffle(chunk) 
        
        # Add to training data
        for text, label in chunk:
            train_x.append(text)
            train_y.append(label)
        
        pos_idx += pos_per_chunk
        neg_idx += neg_per_chunk

    test_x = [posPhrases[i] for i in test_pos] + [negPhrases[n] for n in test_neg]
    test_y = [1]*len(test_pos) + [0]*len(test_neg)

    ethDict = {
        'train': Dataset.from_dict({'label': train_y, 'text': train_x}),
        'test': Dataset.from_dict({'label': test_y, 'text': test_x})
    }
    ethDict = DatasetDict(ethDict)

    stats = {
        'total': {
            'all': len(df),
            'positive': len(pos),
            'negative': len(neg)
        },
        'train_pre_oversampling': {
            'positive': len(train_pos),
            'negative': len(train_neg),
            'total': len(train_pos) + len(train_neg)
        },
        'test_pre_oversampling': {
            'positive': len(test_pos),
            'negative': len(test_neg),
            'total': len(test_pos) + len(test_neg)
        },
        'train_oversampled': {
            'positive': pos_count,
            'negative': neg_count,
            'total': total_examples,
            'chunks': chunks,
            'batches_per_epoch': num_batches
        }
    }

    return ethDict, stats

def freeze_base_layers(model):
    base_model_attr = model.config.model_type
    
    if hasattr(model, base_model_attr):
        for param in getattr(model, base_model_attr).parameters():
            param.requires_grad = False
        print(f"Base layers of {base_model_attr} frozen")
    else:
        print(f"Could not identify base layers for {model.config.model_type}")
    
    return model

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        log_softmax = F.log_softmax(inputs, dim=1)
        
        cross_entropy = F.nll_loss(log_softmax, targets, weight=self.alpha, reduction='none')
        
        probs = torch.exp(log_softmax)
        class_mask = torch.zeros(inputs.size(), device=inputs.device)
        class_mask.scatter_(1, targets.view(-1, 1), 1.)
        
        pt = (probs * class_mask).sum(1)
        
        focal_weight = (1 - pt) ** self.gamma
        
        loss = focal_weight * cross_entropy
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class FocalLossTrainer(Trainer):
    def __init__(self, gamma=2.0, alpha=None, *args, **kwargs):
        super(FocalLossTrainer, self).__init__(*args, **kwargs)
        self.gamma = gamma
        
        if alpha is not None:
            if not isinstance(alpha, torch.Tensor):
                alpha = torch.tensor(alpha)
            self.alpha = alpha
        else:
            self.alpha = None
            
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        loss_fct = FocalLoss(gamma=self.gamma, alpha=self.alpha)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

def compute_metrics(prediction_output):
    y_pred_logits = prediction_output.predictions
    y_true = prediction_output.label_ids

    y_pred_proba = torch.nn.functional.softmax(torch.tensor(y_pred_logits), dim=1).numpy()

    y_pred_class = np.argmax(y_pred_proba, axis=1)

    y_pred_proba_pos = y_pred_proba[:, 1]

    cm = confusion_matrix(y_true, y_pred_class)
    TN, FP, FN, TP = cm.ravel()

    accuracy = accuracy_score(y_true, y_pred_class)
    recall = recall_score(y_true, y_pred_class)
    specificity = TN / (TN + FP)
    precision = precision_score(y_true, y_pred_class)
    npv = TN / (TN + FN)
    f1 = f1_score(y_true, y_pred_class)
    ber = 1 - 0.5 * (recall + specificity)
    roc_auc = roc_auc_score(y_true, y_pred_proba_pos)
    
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba_pos)
    pr_auc = auc(recall_curve, precision_curve)

    return {
        'accuracy': accuracy,
        'recall': recall,        
        'precision': precision,  
        'specificity': specificity,  
        'npv': npv,                  
        'roc_auc': roc_auc,  
        'pr_auc': pr_auc,     
        'f1': f1,    
        'ber': ber  
    }

def create_prediction_df(prediction_output):
    y_pred_logits = prediction_output.predictions
    y_true = prediction_output.label_ids
    y_pred_proba = torch.nn.functional.softmax(torch.tensor(y_pred_logits), dim=1).numpy()
    y_pred_class = np.argmax(y_pred_proba, axis=1)
    y_pred_proba_pos = y_pred_proba[:, 1]
    pred_df = pd.DataFrame({
        'True Label': y_true,
        'Predicted Class': y_pred_class,
        'Predicted Proba (Yes)': y_pred_proba_pos,
        'Raw Prediction 1 (No)': y_pred_logits[:, 0],
        'Raw Prediction 2 (Yes)': y_pred_logits[:, 1]
    })
    return pred_df

def plot_metrics(metrics):
    display_names = {
        'accuracy': 'Accuracy',
        'recall': 'Sensitivity (TPR)',
        'precision': 'Precision (PPV)',
        'specificity': 'Specificity (TNR)',
        'npv': 'NPV',
        'roc_auc': 'ROC AUC',
        'pr_auc': 'PR AUC',
        'f1': 'F1 Score',
        'ber': 'Balanced Error Rate',
    }
    
    items = [(display_names.get(k, k), v) for k, v in metrics.items()]
    labels, values = zip(*items)
    indexes = np.arange(len(labels))
    
    width = 0.8
    plt.figure(figsize=(12, 6))
    bars = plt.bar(indexes, values, width, color='c', label='Metrics', alpha=0.75)
    plt.xlabel('Metrics', fontsize=12)
    plt.ylabel('Values', fontsize=12)
    plt.title('Model Evaluation Metrics', fontsize=14)
    plt.xticks(indexes, labels, rotation=45)
    plt.ylim(0, 1.1)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.4f}', ha='center', color='black', fontsize=10)

    plt.tight_layout()
    return plt.gcf()
    
    
def plot_roc_curve(y_true, y_pred_proba_pos):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba_pos)
    roc_auc = roc_auc_score(y_true, y_pred_proba_pos)

    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic', fontsize=14)
    plt.legend(loc="lower right")
    return plt.gcf()

def plot_training_progression(training_stats):
    if not training_stats:
        print("Warning: Empty training stats provided. Cannot plot progression.")
        return None

    train_logs = [log for log in training_stats if 'loss' in log and 'eval_loss' not in log]
    eval_logs = [log for log in training_stats if 'eval_loss' in log]
    
    if not train_logs and not eval_logs:
        print("Warning: No valid training or evaluation logs found in stats.")
        return None
    
    train_df = pd.DataFrame(train_logs)
    eval_df = pd.DataFrame(eval_logs)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    pos_color = 'green'
    neg_color = 'blue'

    if 'loss' in train_df.columns and 'step' in train_df.columns:
        axes[0, 0].plot(train_df['step'], train_df['loss'], 'o-', color='red', label='Training Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Steps')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, linestyle='--', alpha=0.7)

    if 'eval_loss' in eval_df.columns and 'step' in eval_df.columns:
        axes[0, 1].plot(eval_df['step'], eval_df['eval_loss'], 'o-', color='red', label='Evaluation Loss')
        axes[0, 1].set_title('Evaluation Loss')
        axes[0, 1].set_xlabel('Steps')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, linestyle='--', alpha=0.7)

    if 'eval_f1' in eval_df.columns and 'step' in eval_df.columns:
        axes[0, 2].plot(eval_df['step'], eval_df['eval_f1'], 'o-', color='purple', label='F1 Score')
        axes[0, 2].set_title('F1 Score')
        axes[0, 2].set_xlabel('Steps')
        axes[0, 2].set_ylabel('F1')
        axes[0, 2].set_ylim(0, 1.05)
        axes[0, 2].axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
        axes[0, 2].grid(True, linestyle='--', alpha=0.7)

    has_roc_auc = 'eval_roc_auc' in eval_df.columns and 'step' in eval_df.columns
    has_pr_auc = 'eval_pr_auc' in eval_df.columns and 'step' in eval_df.columns
    
    if has_roc_auc or has_pr_auc:
        if has_roc_auc:
            axes[1, 0].plot(eval_df['step'], eval_df['eval_roc_auc'], 'o-', color='purple', label='ROC AUC')
        if has_pr_auc:
            axes[1, 0].plot(eval_df['step'], eval_df['eval_pr_auc'], 'o-', color='orange', label='PR AUC')
        
        axes[1, 0].set_title('ROC AUC & PR AUC')
        axes[1, 0].set_xlabel('Steps')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].set_ylim(0, 1.05)
        axes[1, 0].axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
        axes[1, 0].legend()
        axes[1, 0].grid(True, linestyle='--', alpha=0.7)

    has_sensitivity = 'eval_recall' in eval_df.columns and 'step' in eval_df.columns
    has_specificity = 'eval_specificity' in eval_df.columns and 'step' in eval_df.columns
    
    if has_sensitivity or has_specificity:
        if has_sensitivity:
            axes[1, 1].plot(eval_df['step'], eval_df['eval_recall'], 'o-', color=pos_color, label='Sensitivity (TPR)')
        if has_specificity:
            axes[1, 1].plot(eval_df['step'], eval_df['eval_specificity'], 'o-', color=neg_color, label='Specificity (TNR)')
        
        axes[1, 1].set_title('Sensitivity & Specificity')
        axes[1, 1].set_xlabel('Steps')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_ylim(0, 1.05)
        axes[1, 1].axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
        axes[1, 1].legend()
        axes[1, 1].grid(True, linestyle='--', alpha=0.7)

    has_precision = 'eval_precision' in eval_df.columns and 'step' in eval_df.columns
    has_npv = 'eval_npv' in eval_df.columns and 'step' in eval_df.columns
    
    if has_precision or has_npv:
        if has_precision:
            axes[1, 2].plot(eval_df['step'], eval_df['eval_precision'], 'o-', color=pos_color, label='Precision (PPV)')
        if has_npv:
            axes[1, 2].plot(eval_df['step'], eval_df['eval_npv'], 'o-', color=neg_color, label='NPV')
        
        axes[1, 2].set_title('Precision & NPV')
        axes[1, 2].set_xlabel('Steps')
        axes[1, 2].set_ylabel('Score')
        axes[1, 2].set_ylim(0, 1.05)
        axes[1, 2].axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
        axes[1, 2].legend()
        axes[1, 2].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    return plt.gcf()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d"
    )
    logger = logging.getLogger('mineth')
    
    args = parse_args()
    config = load_config(args.config)
    config = update_config_with_args(config, args)

    base = config["base"]
    base_path = f"./Model_{base}"
    train_path = config["train_path"]
    val_path = config["val_path"]
    max_length = 200

    os.makedirs(base_path, exist_ok=True)
    os.makedirs(os.path.join(base_path, "prediction"), exist_ok=True)
    
    training_config = config.get("training", {}).copy()
    early_stopping_patience = training_config.pop("early_stopping_patience", 3)
    training_config["output_dir"] = os.path.join(base_path, "training")
    
    seed = training_config.get("seed", 123)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    set_seeds(seed)
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    logger.info("Configuration:")
    logger.info(json.dumps(config, indent=2))

    try:
        training_args = TrainingArguments(**training_config)
    except Exception as e:
        logger.error(f"Error creating TrainingArguments: {e}")
        logger.error(f"Attempted with parameters: {training_config}")
        raise
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    model_type = config["model_type"]
    num_classes = config["num_classes"]
    
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    model = AutoModelForSequenceClassification.from_pretrained(model_type, num_labels=num_classes)
    
    if config.get("freeze_base_layers", False):
        model = freeze_base_layers(model)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    print(f"Using model: {config['model_type']}")

    df = pd.read_table(train_path, encoding='latin-1')
    
    per_device_train_batch_size = training_config.get("per_device_train_batch_size", 16)
    gradient_accumulation_steps = training_config.get("gradient_accumulation_steps", 1)
    
    trestEthDict, data_stats = create_dict(
        df, base, seed=seed,
        batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps
    )
    
    trestEthTokenized = trestEthDict.map(
        lambda examples: tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length), 
        batched=True
    )

    trainer = FocalLossTrainer(
        model=model,
        args=training_args,
        train_dataset=trestEthTokenized['train'],
        eval_dataset=trestEthTokenized['test'],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
    )

    trainer.train()

    trainer.save_model(output_dir=os.path.join(base_path, "trainer"))
    model.save_pretrained(os.path.join(base_path, "model"))
    tokenizer.save_pretrained(os.path.join(base_path, "tokenizer"))
    
    with open(os.path.join(base_path, "training", "data_train.json"), "w") as f:
        json.dump(data_stats, f, indent=2)
    training_stats = trainer.state.log_history
    with open(os.path.join(base_path, "training", "training_stats.json"), "w") as f:
        json.dump(training_stats, f, indent=2)
    plot_training_progression(training_stats)
    plt.savefig(os.path.join(base_path, "training", "training_metrics.png"), dpi=600, bbox_inches="tight")
    
    set_seeds(seed)
    val_df = pd.read_table(val_path, encoding='latin-1')
    val_y = val_df[base].map({'yes': 1, 'no': 0}).astype(int).values
    val_x = list(val_df['SENTENCE'])

    val_stats = {
        'all': len(val_y),
        'positive': int((val_y == 1).sum()),
        'negative': int((val_y == 0).sum())
    }
    with open(os.path.join(base_path, "prediction", "data_val.json"), "w") as f:
        json.dump(val_stats, f, indent=2)

    valEthDict = {
        'val': Dataset.from_dict({'label': val_y, 'text': val_x})
    }
    valEthDict = DatasetDict(valEthDict)

    valEthTokenized = valEthDict.map(
        lambda examples: tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length),
        batched=True
    )
    val_predictions = trainer.predict(valEthTokenized['val'])

    pred_df = create_prediction_df(val_predictions)
    pred_df.to_csv(os.path.join(base_path, "prediction", f"prediction_{base}.txt"), sep='\t', index=False)

    val_metrics = compute_metrics(val_predictions)

    plot_metrics(val_metrics)
    plt.savefig(os.path.join(base_path, "prediction", f"prediction_metrics_{base}.png"), dpi=600)

    plot_roc_curve(pred_df['True Label'], pred_df['Predicted Proba (Yes)'])
    plt.savefig(os.path.join(base_path, "prediction", f"prediction_roc_{base}.png"), dpi=600)

if __name__ == "__main__":
    main()

