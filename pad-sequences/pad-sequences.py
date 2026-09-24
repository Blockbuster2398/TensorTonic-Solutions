import numpy as np

def pad_sequences(seqs: list, pad_value: int = 0, max_len: int | None = None) -> np.ndarray:
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    if not seqs:
        return np.array([], dtype="int").reshape(0, 0)
    padded_seqs = []
    if max_len is None:
        max_len = max([len(seq) for seq in seqs])
    for seq in seqs:
        if len(seq) >= max_len:
            padded_seqs.append(seq[:max_len])
        else:
            padded_seqs.append(seq + [pad_value] * (max_len - len(seq)))
    print(padded_seqs)
    return np.asarray(padded_seqs, dtype="int")