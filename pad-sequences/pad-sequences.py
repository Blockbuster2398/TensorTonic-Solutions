import numpy as np

def pad_sequences(seqs: list, pad_value: int = 0, max_len: int | None = None) -> np.ndarray:
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    
    if max_len is None:
        max_len = 0
        for seq in seqs:
            max_len = max(max_len, len(seq))

    if len(seqs) == 0:
        return np.empty((0, max_len), dtype="int")
    
    padded_sequences = []

    
    for seq in seqs:
        print(f"modified seq: {seq}")
        padded_seq = np.pad(seq, (0, max(0, max_len-len(seq))), 'constant', constant_values=pad_value)
        print(f"modified seq: {padded_seq}")
        padded_sequences.append(padded_seq[:max_len])
        print(f"Ans: {padded_sequences}")
    return np.array(padded_sequences, dtype="int")