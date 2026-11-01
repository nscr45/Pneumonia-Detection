# Model Card: Chest X-ray Pneumonia Classifier

## Summary

A TensorFlow/Keras convolutional neural network packaged as `best_cnn_model.h5`. The Streamlit application uses the model for binary chest X-ray classification and displays **Pneumonia** or **No Pneumonia**.

## Verified deployment details

- Input shape: `128 × 128 × 1` (grayscale).
- Output: one sigmoid value.
- App preprocessing: grayscale conversion, resize to model input dimensions, and pixel scaling by `1/255`.
- App decision threshold: `0.45`.
- Positive-class interpretation in the app: values at or above the threshold are labeled Pneumonia.

These are deployment facts, not proof that the class mapping or threshold is scientifically optimal. Verify both against the original training and validation pipeline.

## Intended use

Educational experimentation and demonstration of a computer-vision inference workflow only.

## Out-of-scope use

- Clinical diagnosis or exclusion of pneumonia.
- Patient triage or treatment decisions.
- Autonomous interpretation of medical images.
- Any claim of clinical performance without appropriate validation and governance.

## Training data and evaluation

Dataset provenance, licensing, cohort composition, data split methodology, class-index mapping and held-out test metrics were not documented in the deployed repository reviewed for this update. They must be supplied from the original research records. Do not infer or fabricate these details.

## Known limitations and risks

- Distribution shift across hospitals, scanners, populations and acquisition protocols may affect predictions.
- The sigmoid output is not necessarily a calibrated probability.
- A threshold of 0.45 requires validation-based justification.
- DICOM pixel interpretation may require modality-specific rescaling and windowing.
- PDF page selection can introduce non-X-ray content into the model.
- The model has not been established as clinically validated.

## Validation checklist

1. Verify that the sigmoid-positive class is Pneumonia.
2. Confirm that inference preprocessing exactly matches training preprocessing.
3. Document the validation procedure used to select the threshold.
4. Evaluate on a held-out, leakage-aware test set.
5. Report confusion matrix, per-class precision/recall/F1, sensitivity, specificity, and ROC-AUC where appropriate.
6. Document dataset source, license, cohort limitations and intended population.

## Safety statement

This model is a research/educational artifact only. Its predictions must not be used to make decisions about patient care. Clinical use would require appropriate expert oversight, external validation, governance and regulatory review.
