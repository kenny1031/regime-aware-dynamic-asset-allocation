import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from src.regime.preprocess import load_regime_features
from src.regime.regime_models import LSTMRegimeClassifier
from src.utils.paths import PROCESSED_DIR
from src.data.build_regime_features import FEATURE_COLUMNS


DEFAULT_LSTM_FEATURES = FEATURE_COLUMNS


def load_hmm_labels(filename: str = "hmm_regime_labels.csv") -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / filename)
    df["date"] = pd.to_datetime(df["date"])
    return df


def build_supervised_dataset(
    feature_cols: list[str] | None = None,
    feature_filename: str = "regime_features.csv",
    label_filename: str = "hmm_regime_labels.csv",
) -> pd.DataFrame:
    if feature_cols is None:
        feature_cols = DEFAULT_LSTM_FEATURES

    features_df = load_regime_features(feature_filename)
    labels_df = load_hmm_labels(label_filename)

    merged = features_df.merge(
        labels_df[["date", "regime", "regime_name"]],
        on="date",
        how="inner"
    )

    merged = merged[["date"] + feature_cols + ["regime", "regime_name"]].copy()
    merged = merged.dropna().sort_values("date").reset_index(drop=True)

    return merged


def build_sequence_dataset(X, y, dates, seq_len: int = 12):
    """
    Build many-to-one sequence dataset:
        (X_{t-seq_len+1}, ..., X_t) -> y_t
    """
    X_seq, y_seq, dates_seq = [], [], []

    for i in range(seq_len - 1, len(X)):
        X_seq.append(X[i - seq_len + 1:i + 1])
        y_seq.append(y[i])
        dates_seq.append(dates[i])

    return np.array(X_seq), np.array(y_seq), np.array(dates_seq)


class SequenceDataset(Dataset):
    def __init__(self, X_seq: np.ndarray, y_seq: np.ndarray):
        self.X_seq = torch.tensor(X_seq, dtype=torch.float32)
        self.y_seq = torch.tensor(y_seq, dtype=torch.long)

    def __len__(self):
        return len(self.X_seq)

    def __getitem__(self, idx):
        return self.X_seq[idx], self.y_seq[idx]


def time_based_split_sequence(
    X_seq: np.ndarray,
    y_seq: np.ndarray,
    dates_seq: np.ndarray,
    train_frac: float = 0.7,
):
    split_idx = int(len(X_seq) * train_frac)

    X_train = X_seq[:split_idx]
    y_train = y_seq[:split_idx]
    dates_train = dates_seq[:split_idx]

    X_test = X_seq[split_idx:]
    y_test = y_seq[split_idx:]
    dates_test = dates_seq[split_idx:]

    return X_train, y_train, dates_train, X_test, y_test, dates_test


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    total_count = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        batch_size = X_batch.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size

    return total_loss / total_count


@torch.no_grad()
def evaluate_model(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_count = 0

    all_preds = []
    all_probs = []
    all_true = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        batch_size = X_batch.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size

        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_true.append(y_batch.cpu().numpy())

    avg_loss = total_loss / total_count
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)

    return avg_loss, y_true, y_pred, y_prob


def build_prediction_table(
    dates_test: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    n_classes: int = 3,
) -> pd.DataFrame:
    name_map = {
        0: "Neutral",
        1: "Risk-On",
        2: "Risk-Off",
    }

    out = pd.DataFrame({
        "date": pd.to_datetime(dates_test),
        "true_regime": y_true,
        "pred_regime": y_pred,
    })

    out["true_regime_name"] = out["true_regime"].map(name_map)
    out["pred_regime_name"] = out["pred_regime"].map(name_map)

    for k in range(n_classes):
        out[f"prob_regime_{k}"] = y_prob[:, k]

    return out


def main():
    # Config
    feature_cols = DEFAULT_LSTM_FEATURES
    seq_len = 12
    train_frac = 0.7
    batch_size = 32
    epochs = 30
    learning_rate = 1e-3
    hidden_size = 64
    num_layers = 2
    dropout = 0.2
    num_classes = 3
    random_seed = 42

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Load merged supervised dataset
    df = build_supervised_dataset(feature_cols=feature_cols)

    X = df[feature_cols].values
    y = df["regime"].values
    dates = df["date"].values

    # 1) build sequences from raw features
    # 2) split
    # 3) fit scaler on train rows only
    X_seq_raw, y_seq, dates_seq = build_sequence_dataset(X, y, dates, seq_len=seq_len)

    X_train_raw, y_train, dates_train, X_test_raw, y_test, dates_test = time_based_split_sequence(
        X_seq_raw, y_seq, dates_seq, train_frac=train_frac
    )

    # Fit scaler only on training data (flatten sequence dimension)
    scaler = StandardScaler()
    X_train_2d = X_train_raw.reshape(-1, X_train_raw.shape[-1])
    scaler.fit(X_train_2d)

    def transform_sequences(X_seq_raw):
        n, l, d = X_seq_raw.shape
        X_2d = X_seq_raw.reshape(-1, d)
        X_scaled = scaler.transform(X_2d)
        return X_scaled.reshape(n, l, d)

    X_train = transform_sequences(X_train_raw)
    X_test = transform_sequences(X_test_raw)

    train_dataset = SequenceDataset(X_train, y_train)
    test_dataset = SequenceDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = LSTMRegimeClassifier(
        input_size=len(feature_cols),
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("\n=== Training LSTM Regime Classifier ===")
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, y_true_eval, y_pred_eval, y_prob_eval = evaluate_model(model, test_loader, criterion, device)

        acc = accuracy_score(y_true_eval, y_pred_eval)
        macro_f1 = f1_score(y_true_eval, y_pred_eval, average="macro")

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Acc: {acc:.4f} | "
            f"Macro F1: {macro_f1:.4f}"
        )

    # Final evaluation
    test_loss, y_true, y_pred, y_prob = evaluate_model(model, test_loader, criterion, device)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["true_0", "true_1", "true_2"],
        columns=["pred_0", "pred_1", "pred_2"],
    )

    report = classification_report(y_true, y_pred, digits=4)

    pred_df = build_prediction_table(
        dates_test=dates_test,
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        n_classes=num_classes,
    )

    pred_df.to_csv(PROCESSED_DIR / "lstm_regime_predictions.csv", index=False)
    cm_df.to_csv(PROCESSED_DIR / "lstm_regime_confusion_matrix.csv", index=True)

    with open(PROCESSED_DIR / "lstm_regime_classification_report.txt", "w") as f:
        f.write(report)

    print(f"\nLSTM predictions saved to {PROCESSED_DIR / 'lstm_regime_predictions.csv'}")
    print(f"LSTM confusion matrix saved to {PROCESSED_DIR / 'lstm_regime_confusion_matrix.csv'}")
    print(f"LSTM classification report saved to {PROCESSED_DIR / 'lstm_regime_classification_report.txt'}")

    print("\n=== Train/Test Split ===")
    print(f"Train sequences: {len(X_train)}")
    print(f"Test sequences:  {len(X_test)}")
    print(f"Train start: {pd.to_datetime(dates_train.min()).date()}")
    print(f"Train end: {pd.to_datetime(dates_train.max()).date()}")
    print(f"Test start: {pd.to_datetime(dates_test.min()).date()}")
    print(f"Test end: {pd.to_datetime(dates_test.max()).date()}")

    print("\n=== LSTM Regime Classification Metrics ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    print("\n=== Confusion Matrix ===")
    print(cm_df)

    print("\n=== Classification Report ===")
    print(report)

    print("\n=== Prediction Sample ===")
    print(pred_df.head(15))


if __name__ == "__main__":
    main()