# Model Card: Pneumonia Detection CNN

## What this model does

I trained and saved a CNN model in `best_cnn_model.h5` for a chest X-ray classification project. The application uses it to classify an uploaded image as **Pneumonia** or **No Pneumonia**.

## Model details

From the saved model file, I confirmed the following:

- Input shape: `128 × 128 × 1` (grayscale image).
- Output layer: one sigmoid value.
- Layers: two convolution and max-pooling blocks, followed by a flatten layer, a dense layer, dropout and the final sigmoid output.
- Threshold currently used in the app: `0.45`.

For prediction, my app converts the image to grayscale, resizes it to the model's input dimensions and scales pixel values by `1/255`. I still need to compare this preprocessing and the positive-class mapping with my original training notebook to make sure they match exactly.

## Intended use

I built this model as a learning and portfolio project to practise image classification and deploy a trained CNN through Streamlit.

It is not intended for clinical diagnosis, patient triage or treatment decisions.

## Dataset and evaluation

I still need to add the dataset source, licensing details, dataset split and class mapping from my original project records. I also need to evaluate the saved model on a separate test set before publishing performance metrics.

I have not included accuracy, precision, recall, F1-score, sensitivity, specificity or ROC-AUC here because I have not yet verified those results against the actual evaluation output.

## Current limitations

- I need to confirm that the model's sigmoid output represents the Pneumonia class.
- The `0.45` threshold is the value currently set in the app; it has not been justified here using a documented validation analysis.
- A sigmoid output should not automatically be treated as a calibrated medical probability.
- Model performance may change when images come from different hospitals, scanners or patient populations.
- DICOM images may need modality-specific rescaling and windowing. The current image conversion should be tested with representative DICOM files.
- For PDF uploads, the user has to select a page that actually contains a chest X-ray.
- I have not established this model as clinically validated.

## Work still to do

1. Check the model's class order against the training notebook.
2. Confirm that inference preprocessing matches the training pipeline.
3. Evaluate the model on a separate test set.
4. Add a confusion matrix and relevant classification metrics.
5. Review the threshold using validation results.
6. Record the dataset source, license and any relevant acknowledgements.

## Medical disclaimer

This is an educational project, not a medical device. Its predictions should not be used to diagnose or rule out pneumonia or to make healthcare decisions. Anyone with health concerns should consult a qualified healthcare professional.
