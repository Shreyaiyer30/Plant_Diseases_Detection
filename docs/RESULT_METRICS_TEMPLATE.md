# PlantCure - Result Metrics Template

Use this template in your report/results chapter after model training.

## 1) Experiment Setup

- Date:
- Dataset path:
- Number of classes used:
- Train/validation split:
- Image size:
- Batch size:
- Epochs:
- Fine-tune: Yes/No
- Label smoothing:
- Hardware (CPU/GPU, RAM):

## 2) Dataset Summary

| Class Name | Train Count | Validation Count |
|---|---:|---:|
| Class 1 |  |  |
| Class 2 |  |  |
| ... |  |  |
| Total |  |  |

## 3) Final Training Metrics

| Metric | Value |
|---|---:|
| Final Train Accuracy |  |
| Final Validation Accuracy |  |
| Final Train Loss |  |
| Final Validation Loss |  |
| Best Validation Accuracy |  |
| Epoch of Best Validation Accuracy |  |

## 4) Per-Class Performance

| Class Name | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| Class 1 |  |  |  |  |
| Class 2 |  |  |  |  |
| ... |  |  |  |  |

## 5) Confusion Matrix

- Add confusion matrix image here.
- Mention strongest and weakest class separations.

## 6) Real-World Validation (App Testing)

| Test Case | Input Type | Expected | Actual | Pass/Fail |
|---|---|---|---|---|
| Healthy leaf | Upload | Healthy class |  |  |
| Diseased leaf | Upload | Correct disease |  |  |
| Camera capture | Live camera | Correct class |  |  |
| Non-leaf image | Hand/object | Invalid image rejection |  |  |
| Blurry image | Upload/camera | Low confidence reject |  |  |

## 7) Inference Performance

| Metric | Value |
|---|---:|
| Avg API prediction time (s) |  |
| Avg page response time (s) |  |
| Model size (MB) |  |

## 8) Error Analysis

- Common misclassifications:
- Root causes:
- Steps taken to improve:

## 9) Conclusion

- Final achieved validation accuracy:
- Practical usefulness:
- Key limitations:
- Future improvements:
