"""
LSTM model สำหรับพยากรณ์ VIX — Tier 3

ใช้ PyTorch (มีอยู่แล้วใน deps ผ่าน transformers)
ห่อด้วย sklearn-like interface (fit/predict/predict_proba)
เพื่อให้พลั่ก-อิน-แพล็ก กับ model registry ของเรา

⚠️ Honest note: LSTM ต้องการข้อมูลเยอะกว่า tree models
   บนข้อมูลของเรา (~1,200 วัน) มัก underperform XGBoost
   แต่ใส่มาเพื่อ:
   1. แสดง breadth (รู้ deep learning)
   2. เป็น baseline สำหรับ comparison
   3. พร้อมขยายเมื่อมีข้อมูลเยอะกว่านี้ (intraday, more years)
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class _LSTMNet(nn.Module):
    """Simple 1-layer LSTM + dense head"""
    def __init__(self, input_size, hidden_size=32, output_size=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, dropout=0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])  # take last timestep
        return self.fc(out)


class LSTMRegressor:
    """sklearn-style wrapper รอบ LSTM"""

    def __init__(self, seq_length=10, hidden_size=32, epochs=50,
                 lr=1e-3, batch_size=32, random_state=42):
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.random_state = random_state
        self.model = None
        self.feature_mean = None
        self.feature_std = None

    def _prepare_sequences(self, X, y=None):
        """แปลง [n_samples, n_features] → [n_samples, seq_length, n_features]"""
        X = np.asarray(X, dtype=np.float32)
        seqs = []
        targets = []
        for i in range(len(X) - self.seq_length):
            seqs.append(X[i:i + self.seq_length])
            if y is not None:
                targets.append(y[i + self.seq_length - 1])

        seqs = np.stack(seqs) if seqs else np.empty((0, self.seq_length, X.shape[1]))
        if y is not None:
            return seqs, np.asarray(targets, dtype=np.float32)
        return seqs

    def fit(self, X, y):
        torch.manual_seed(self.random_state)

        # Normalize features (LSTM ต้องการ scaled input)
        X = np.asarray(X, dtype=np.float32)
        self.feature_mean = X.mean(axis=0)
        self.feature_std = X.std(axis=0) + 1e-8
        X_scaled = (X - self.feature_mean) / self.feature_std

        y = np.asarray(y, dtype=np.float32)

        X_seq, y_seq = self._prepare_sequences(X_scaled, y)
        if len(X_seq) == 0:
            raise ValueError(f"Not enough data for sequences (need > {self.seq_length})")

        dataset = TensorDataset(torch.from_numpy(X_seq), torch.from_numpy(y_seq).unsqueeze(1))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = _LSTMNet(input_size=X.shape[1], hidden_size=self.hidden_size)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        self.model.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = self.model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()

        return self

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("Model not fitted")
        X = np.asarray(X, dtype=np.float32)
        X_scaled = (X - self.feature_mean) / self.feature_std

        # ถ้า X สั้นกว่า seq_length → pad ด้วยค่าซ้ำ (สำหรับ inference จุดเดียว)
        if len(X_scaled) < self.seq_length:
            pad_needed = self.seq_length - len(X_scaled)
            padding = np.repeat(X_scaled[:1], pad_needed, axis=0)
            X_scaled = np.vstack([padding, X_scaled])

        X_seq = self._prepare_sequences(X_scaled)
        # ถ้ายังว่าง (กรณี X = 1 row) → ใช้ทั้ง padded window
        if len(X_seq) == 0:
            X_seq = X_scaled[np.newaxis, ...]

        self.model.eval()
        with torch.no_grad():
            preds = self.model(torch.from_numpy(X_seq)).numpy().flatten()
        # คืนค่าให้มี length เท่ากับ input ด้วย repeat ของ prediction แรก (สำหรับ rows ที่ไม่มี history พอ)
        n_input = len(X)
        n_pred = len(preds)
        if n_pred < n_input:
            pad = np.full(n_input - n_pred, preds[0])
            preds = np.concatenate([pad, preds])
        return preds

    def score(self, X, y):
        """R² — สำหรับ sklearn compat"""
        from sklearn.metrics import r2_score
        y_pred = self.predict(X)
        return r2_score(np.asarray(y), y_pred)
