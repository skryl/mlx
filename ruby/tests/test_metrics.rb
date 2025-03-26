require_relative 'mlx_test_case'

class TestMetrics < MLXTestCase
  def test_accuracy
    # Test binary classification
    y_true = MLX.array([0, 1, 0, 1, 0])
    y_pred = MLX.array([0, 1, 1, 1, 0])
    acc = MLX.metrics.accuracy(y_true, y_pred)
    assert_equal 0.8, acc.item
    
    # Test multiclass classification
    y_true = MLX.array([0, 1, 2, 1, 0])
    y_pred = MLX.array([0, 1, 1, 1, 0])
    acc = MLX.metrics.accuracy(y_true, y_pred)
    assert_equal 0.8, acc.item
    
    # Test with logits
    logits = MLX.array([
      [0.9, 0.1, 0.0],
      [0.2, 0.7, 0.1],
      [0.1, 0.8, 0.1],
      [0.0, 0.9, 0.1],
      [0.8, 0.1, 0.1]
    ])
    y_true = MLX.array([0, 1, 2, 1, 0])
    acc = MLX.metrics.accuracy(y_true, MLX.argmax(logits, axis: 1))
    assert_equal 0.6, acc.item
    
    # Test with threshold for binary classification
    y_prob = MLX.array([0.9, 0.6, 0.4, 0.7, 0.1])
    y_true = MLX.array([1, 1, 0, 1, 0])
    
    # Default threshold 0.5
    y_pred = (y_prob >= 0.5).astype(MLX.int32)
    acc = MLX.metrics.accuracy(y_true, y_pred)
    assert_equal 0.8, acc.item
    
    # Custom threshold 0.6
    y_pred = (y_prob >= 0.6).astype(MLX.int32)
    acc = MLX.metrics.accuracy(y_true, y_pred)
    assert_equal 0.6, acc.item
  end
  
  def test_precision
    # Test binary precision
    y_true = MLX.array([0, 1, 0, 1, 0])
    y_pred = MLX.array([1, 1, 0, 1, 0])
    prec = MLX.metrics.precision(y_true, y_pred)
    assert_equal 0.67, (prec.item * 100).round / 100.0 # Rounded to handle floating point
    
    # Test multiclass precision (macro)
    y_true = MLX.array([0, 1, 2, 1, 0])
    y_pred = MLX.array([0, 1, 1, 1, 0])
    prec = MLX.metrics.precision(y_true, y_pred, average: 'macro')
    # Class 0: 1.0, Class 1: 0.67, Class 2: 0.0 -> avg = 0.56
    assert_in_delta 0.56, prec.item, 0.01
    
    # Test multiclass precision (micro)
    prec = MLX.metrics.precision(y_true, y_pred, average: 'micro')
    assert_equal 0.8, prec.item
    
    # Test with binary labels and class index
    y_true = MLX.array([0, 1, 0, 1, 0])
    y_pred = MLX.array([1, 1, 0, 0, 0])
    prec = MLX.metrics.precision(y_true, y_pred, pos_label: 1)
    assert_equal 0.5, prec.item
    
    # Test handling of empty predictions
    y_true = MLX.array([0, 0, 0, 0, 0])
    y_pred = MLX.array([0, 0, 0, 0, 0])
    
    # In this case, precision for class 1 should be undefined or 0
    # Different libraries handle this differently, so we'll check for both possibilities
    prec = MLX.metrics.precision(y_true, y_pred, pos_label: 1)
    assert prec.item == 0.0 || Float::NAN == prec.item
  end
  
  def test_recall
    # Test binary recall
    y_true = MLX.array([0, 1, 0, 1, 0])
    y_pred = MLX.array([1, 1, 0, 0, 0])
    rec = MLX.metrics.recall(y_true, y_pred)
    assert_equal 0.5, rec.item
    
    # Test multiclass recall (macro)
    y_true = MLX.array([0, 1, 2, 1, 0])
    y_pred = MLX.array([0, 1, 1, 1, 0])
    rec = MLX.metrics.recall(y_true, y_pred, average: 'macro')
    # Class 0: 1.0, Class 1: 1.0, Class 2: 0.0 -> avg = 0.67
    assert_in_delta 0.67, rec.item, 0.01
    
    # Test multiclass recall (micro)
    rec = MLX.metrics.recall(y_true, y_pred, average: 'micro')
    assert_equal 0.8, rec.item
    
    # Test with binary labels and class index
    y_true = MLX.array([0, 1, 0, 1, 0])
    y_pred = MLX.array([0, 0, 0, 1, 1])
    rec = MLX.metrics.recall(y_true, y_pred, pos_label: 1)
    assert_equal 0.5, rec.item
    
    # Test handling of empty true positives
    y_true = MLX.array([0, 0, 0, 0, 0])
    y_pred = MLX.array([1, 1, 0, 0, 0])
    
    # In this case, recall for class 1 should be undefined or 0
    rec = MLX.metrics.recall(y_true, y_pred, pos_label: 1)
    assert rec.item == 0.0 || Float::NAN == rec.item
  end
  
  def test_f1_score
    # Test binary f1
    y_true = MLX.array([0, 1, 0, 1, 0])
    y_pred = MLX.array([1, 1, 0, 0, 0])
    f1 = MLX.metrics.f1_score(y_true, y_pred)
    assert_in_delta 0.5, f1.item, 0.01
    
    # Test multiclass f1 (macro)
    y_true = MLX.array([0, 1, 2, 1, 0])
    y_pred = MLX.array([0, 1, 1, 1, 0])
    f1 = MLX.metrics.f1_score(y_true, y_pred, average: 'macro')
    # Class 0: 1.0, Class 1: 0.8, Class 2: 0.0 -> avg = 0.6
    assert_in_delta 0.6, f1.item, 0.01
    
    # Test multiclass f1 (micro)
    f1 = MLX.metrics.f1_score(y_true, y_pred, average: 'micro')
    assert_equal 0.8, f1.item
    
    # Test weighted average
    f1 = MLX.metrics.f1_score(y_true, y_pred, average: 'weighted')
    # Class weights: [2/5, 2/5, 1/5] -> weighted avg = 0.76
    assert_in_delta 0.76, f1.item, 0.01
    
    # Test with binary labels and class index
    y_true = MLX.array([0, 1, 0, 1, 0])
    y_pred = MLX.array([0, 0, 0, 1, 1])
    f1 = MLX.metrics.f1_score(y_true, y_pred, pos_label: 1)
    assert_equal 0.5, f1.item
  end
  
  def test_precision_recall_fscore_support
    # Test all metrics together
    y_true = MLX.array([0, 1, 2, 0, 1, 2])
    y_pred = MLX.array([0, 2, 1, 0, 0, 1])
    
    precision, recall, f1, support = MLX.metrics.precision_recall_fscore_support(y_true, y_pred)
    
    # Check shapes
    assert_equal [3], precision.shape
    assert_equal [3], recall.shape
    assert_equal [3], f1.shape
    assert_equal [3], support.shape
    
    # Check support values
    assert MLX.array_equal(support, MLX.array([2, 2, 2]))
    
    # Check precision values
    # Class 0: 2/3, Class 1: 0/2, Class 2: 0/1
    assert_in_delta 0.67, precision[0].item, 0.01
    assert_equal 0.0, precision[1].item
    assert_equal 0.0, precision[2].item
    
    # Check recall values
    # Class 0: 2/2, Class 1: 0/2, Class 2: 0/2
    assert_equal 1.0, recall[0].item
    assert_equal 0.0, recall[1].item
    assert_equal 0.0, recall[2].item
    
    # Check F1 values
    assert_in_delta 0.8, f1[0].item, 0.01
    assert_equal 0.0, f1[1].item
    assert_equal 0.0, f1[2].item
    
    # Test with macro average
    precision, recall, f1, _ = MLX.metrics.precision_recall_fscore_support(
      y_true, y_pred, average: 'macro'
    )
    
    # Check shapes for averaged values
    assert_equal [], precision.shape  # scalar
    assert_equal [], recall.shape     # scalar
    assert_equal [], f1.shape         # scalar
    
    # Check averaged values
    assert_in_delta 0.22, precision.item, 0.01  # (0.67 + 0 + 0) / 3
    assert_in_delta 0.33, recall.item, 0.01     # (1.0 + 0 + 0) / 3
    assert_in_delta 0.27, f1.item, 0.01         # (0.8 + 0 + 0) / 3
  end
  
  def test_mean_squared_error
    y_true = MLX.array([3.0, -0.5, 2.0, 7.0])
    y_pred = MLX.array([2.5, 0.0, 2.0, 8.0])
    
    mse = MLX.metrics.mean_squared_error(y_true, y_pred)
    # (0.5^2 + 0.5^2 + 0^2 + 1^2) / 4 = 0.375
    assert_equal 0.375, mse.item
    
    # Test with squared=False for RMSE
    rmse = MLX.metrics.mean_squared_error(y_true, y_pred, squared: false)
    assert_in_delta 0.612, rmse.item, 0.001  # √0.375 ≈ 0.612
    
    # Test with sample weights
    weights = MLX.array([1.0, 1.0, 2.0, 0.5])
    mse = MLX.metrics.mean_squared_error(y_true, y_pred, sample_weight: weights)
    # (0.5^2*1 + 0.5^2*1 + 0^2*2 + 1^2*0.5) / 4.5 = 0.333
    assert_in_delta 0.333, mse.item, 0.001
  end
  
  def test_mean_absolute_error
    y_true = MLX.array([3.0, -0.5, 2.0, 7.0])
    y_pred = MLX.array([2.5, 0.0, 2.0, 8.0])
    
    mae = MLX.metrics.mean_absolute_error(y_true, y_pred)
    # (0.5 + 0.5 + 0 + 1) / 4 = 0.5
    assert_equal 0.5, mae.item
    
    # Test with sample weights
    weights = MLX.array([1.0, 1.0, 2.0, 0.5])
    mae = MLX.metrics.mean_absolute_error(y_true, y_pred, sample_weight: weights)
    # (0.5*1 + 0.5*1 + 0*2 + 1*0.5) / 4.5 = 0.333
    assert_in_delta 0.333, mae.item, 0.001
  end
  
  def test_r2_score
    y_true = MLX.array([3.0, -0.5, 2.0, 7.0])
    y_pred = MLX.array([2.5, 0.0, 2.0, 8.0])
    
    r2 = MLX.metrics.r2_score(y_true, y_pred)
    # 1 - sum((y_true - y_pred)^2) / sum((y_true - mean(y_true))^2)
    # mean(y_true) = 2.875
    # 1 - 1.5 / 36.875 = 0.9593
    assert_in_delta 0.959, r2.item, 0.001
    
    # Test with sample weights
    weights = MLX.array([1.0, 1.0, 2.0, 0.5])
    r2 = MLX.metrics.r2_score(y_true, y_pred, sample_weight: weights)
    assert r2.item < 1.0 && r2.item > 0.9  # Should be similar but not identical
    
    # Test with perfect prediction
    r2 = MLX.metrics.r2_score(y_true, y_true)
    assert_equal 1.0, r2.item
    
    # Test with worst case (predicting the mean)
    mean = MLX.mean(y_true)
    y_pred_mean = MLX.full_like(y_true, mean)
    r2 = MLX.metrics.r2_score(y_true, y_pred_mean)
    assert_in_delta 0.0, r2.item, 0.001
  end
  
  def test_binary_confusion_matrix
    y_true = MLX.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = MLX.array([0, 1, 0, 0, 1, 1, 1, 1])
    
    cm = MLX.metrics.confusion_matrix(y_true, y_pred)
    
    # Expected confusion matrix:
    # [2, 2]  # TN, FP
    # [1, 3]  # FN, TP
    expected = MLX.array([[2, 2], [1, 3]])
    assert MLX.array_equal(cm, expected)
    
    # Test with custom labels
    cm = MLX.metrics.confusion_matrix(y_true, y_pred, labels: [1, 0])
    expected = MLX.array([[3, 1], [2, 2]])
    assert MLX.array_equal(cm, expected)
    
    # Test with normalization
    cm = MLX.metrics.confusion_matrix(y_true, y_pred, normalize: 'true')
    # Normalized by rows (true labels):
    # [2/4, 2/4]  # TN, FP for class 0
    # [1/4, 3/4]  # FN, TP for class 1
    expected = MLX.array([[0.5, 0.5], [0.25, 0.75]])
    assert MLX.allclose(cm, expected)
  end
  
  def test_multiclass_confusion_matrix
    y_true = MLX.array([0, 1, 2, 0, 1, 2, 0, 2])
    y_pred = MLX.array([0, 2, 1, 0, 1, 2, 2, 2])
    
    cm = MLX.metrics.confusion_matrix(y_true, y_pred)
    
    # Expected confusion matrix:
    # [2, 0, 1]  # TP/Class 0, FP to Class 1, FP to Class 2
    # [0, 1, 1]  # FP to Class 0, TP/Class 1, FP to Class 2
    # [0, 1, 2]  # FP to Class 0, FP to Class 1, TP/Class 2
    expected = MLX.array([[2, 0, 1], [0, 1, 1], [0, 1, 2]])
    assert MLX.array_equal(cm, expected)
    
    # Test with normalization by all samples
    cm = MLX.metrics.confusion_matrix(y_true, y_pred, normalize: 'all')
    # Sum of all entries is 1.0
    assert_in_delta 1.0, MLX.sum(cm).item, 0.001
    
    # Test with specific labels subset
    cm = MLX.metrics.confusion_matrix(y_true, y_pred, labels: [0, 2])
    expected = MLX.array([[2, 1], [0, 2]])
    assert MLX.array_equal(cm, expected)
  end
  
  def test_precision_at_k
    # Test precision@k for multi-label classification
    y_true = MLX.array([
      [1, 0, 1], 
      [0, 1, 1], 
      [1, 1, 0]
    ])
    
    # Prediction scores (higher = more likely)
    y_score = MLX.array([
      [0.9, 0.2, 0.8],  # Top 2: [0, 2]
      [0.3, 0.7, 0.6],  # Top 2: [1, 2]
      [0.7, 0.8, 0.3]   # Top 2: [0, 1]
    ])
    
    # Calculate precision@k=2 manually:
    # Example 1: 2 correct in top 2 -> precision = 1.0
    # Example 2: 2 correct in top 2 -> precision = 1.0
    # Example 3: 2 correct in top 2 -> precision = 1.0
    # Average: 1.0
    
    # Get top k predictions
    k = 2
    top_k_preds = MLX.argsort(-y_score, axis: 1)[:, 0:k]
    
    # Convert to binary prediction matrix
    y_pred = MLX.zeros_like(y_true)
    
    # We need to manually set the top-k predictions to 1
    for i in 0...y_pred.shape[0]
      for j in 0...k
        y_pred[i, top_k_preds[i, j]] = 1
      end
    end
    
    # Calculate precision for each sample
    sample_precision = MLX.empty(y_true.shape[0])
    for i in 0...y_true.shape[0]
      true_positives = MLX.sum(y_true[i] & y_pred[i])
      predicted_positives = MLX.sum(y_pred[i])
      sample_precision[i] = true_positives / MLX.maximum(predicted_positives, 1e-15)
    end
    
    precision_at_k = MLX.mean(sample_precision)
    assert_equal 1.0, precision_at_k.item
    
    # Now test with less perfect predictions
    y_score = MLX.array([
      [0.9, 0.8, 0.3],  # Top 2: [0, 1], only 1 correct
      [0.3, 0.7, 0.9],  # Top 2: [2, 1], both correct
      [0.7, 0.3, 0.8]   # Top 2: [2, 0], only 1 correct
    ])
    
    # Recalculate with new predictions
    top_k_preds = MLX.argsort(-y_score, axis: 1)[:, 0:k]
    y_pred = MLX.zeros_like(y_true)
    
    for i in 0...y_pred.shape[0]
      for j in 0...k
        y_pred[i, top_k_preds[i, j]] = 1
      end
    end
    
    for i in 0...y_true.shape[0]
      true_positives = MLX.sum(y_true[i] & y_pred[i])
      predicted_positives = MLX.sum(y_pred[i])
      sample_precision[i] = true_positives / MLX.maximum(predicted_positives, 1e-15)
    end
    
    precision_at_k = MLX.mean(sample_precision)
    # (0.5 + 1.0 + 0.5) / 3 = 0.667
    assert_in_delta 0.667, precision_at_k.item, 0.001
  end
  
  def test_roc_auc_score
    y_true = MLX.array([0, 0, 1, 1])
    y_score = MLX.array([0.1, 0.4, 0.35, 0.8])
    
    auc = MLX.metrics.roc_auc_score(y_true, y_score)
    # The AUC for this case should be 0.75
    assert_in_delta 0.75, auc.item, 0.01
    
    # Test with binary classification scores (probabilities for class 1)
    y_score = MLX.array([
      [0.9, 0.1],   # Prob for class 0, class 1
      [0.6, 0.4],   # Prob for class 0, class 1
      [0.65, 0.35], # Prob for class 0, class 1
      [0.2, 0.8]    # Prob for class 0, class 1
    ])
    auc = MLX.metrics.roc_auc_score(y_true, y_score[:, 1])
    assert_in_delta 0.75, auc.item, 0.01
    
    # Test with multiclass case
    y_true = MLX.array([0, 1, 2, 0, 1, 2])
    y_score = MLX.array([
      [0.9, 0.05, 0.05],   # Class 0 scores
      [0.1, 0.8, 0.1],     # Class 1 scores
      [0.1, 0.2, 0.7],     # Class 2 scores
      [0.8, 0.15, 0.05],   # Class 0 scores
      [0.2, 0.7, 0.1],     # Class 1 scores
      [0.3, 0.2, 0.5]      # Class 2 scores
    ])
    
    # Test with multi-class AUC (one-vs-rest)
    auc = MLX.metrics.roc_auc_score(y_true, y_score, average: 'macro', multi_class: 'ovr')
    # Expected to be high since predictions align well with true classes
    assert auc.item > 0.8
  end
end 