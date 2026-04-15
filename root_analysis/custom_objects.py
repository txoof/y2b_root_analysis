def f1(y_true, y_pred):
    """Compute the F1 score (harmonic mean of precision and recall) for binary classification.
    This function operates on Keras backend tensors and computes precision and recall
    from rounded/clipped binary values before combining them into the F1 score.
    True positives, predicted positives and actual positives are computed by
    rounding clipped values to {0, 1}, and a small epsilon (K.epsilon()) is added
    to denominators to avoid division by zero.
    Args:
        y_true: Tensor. Ground-truth labels. Expected shape (batch_size, ...) and
            values are treated as binary after clipping/rounding.
        y_pred: Tensor. Predicted values (e.g. probabilities or logits). Same shape
            as y_true; values are clipped and rounded to produce binary predictions.
    Returns:
        Tensor. Scalar tensor representing the F1 score in the range [0, 1].
    Notes:
        - Implemented using Keras backend (K) operations; intended for use as a
          Keras metric or loss component.
        - The implementation computes:
            TP = sum(round(clip(y_true * y_pred, 0, 1)))
            Predicted = sum(round(clip(y_pred, 0, 1)))
            Actual = sum(round(clip(y_true, 0, 1)))
          then precision = TP / (Predicted + eps), recall = TP / (Actual + eps),
          and F1 = 2 * (precision * recall) / (precision + recall + eps).
    """

    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = TP / (Positives+K.epsilon())
        return recall
    
    def precision_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = TP / (Pred_Positives+K.epsilon())
        return precision
    
    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)
    
    return 2*((precision*recall)/(precision+recall+K.epsilon()))