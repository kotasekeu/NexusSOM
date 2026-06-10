About Dataset
Description:
Breast cancer is the most common cancer amongst women in the world. It accounts for 25% of all cancer cases, and affected over 2.1 Million people in 2015 alone. It starts when cells in the breast begin to grow out of control. These cells usually form tumors that can be seen via X-ray or felt as lumps in the breast area.

The key challenges against it’s detection is how to classify tumors into malignant (cancerous) or benign(non cancerous). We ask you to complete the analysis of classifying these tumors using machine learning (with SVMs) and the Breast Cancer Wisconsin (Diagnostic) Dataset.

Acknowledgements:
This dataset has been referred from Kaggle.

Objective:
Understand the Dataset & cleanup (if required).
Build classification models to predict whether the cancer type is Malignant or Benign.
Also fine-tune the hyperparameters & compare the evaluation metrics of various classification algorithms.

id: A unique identifier for each tumor record.

diagnosis: The diagnosis of the tumor, where M stands for malignant and B stands for benign.

radius_mean: The mean of the distances from the center of the tumor to the perimeter. It reflects the average size of the tumor.
texture_mean: The mean of the standard deviation of gray-scale values in the tumor image. It measures the smoothness or coarseness of the tumor's texture.
perimeter_mean: The mean of the perimeter lengths of the tumor's boundary. It reflects the average circumference of the tumor.
area_mean: The mean of the area occupied by the tumor. It represents the average size of the tumor's surface area.
smoothness_mean: The mean of local variation in radius lengths, measuring how smooth the tumor's surface is.
compactness_mean: The mean of the compactness of the tumor. It indicates how compact or round the tumor is.
concavity_mean: The mean of the severity of concave portions of the tumor's boundary. It reflects the degree of inward curving of the tumor's edges.
concave points_mean: The mean of the number of concave portions of the tumor's boundary. It counts the number of indentations or concave points on the tumor's edge.
symmetry_mean: The mean of the symmetry of the tumor. It measures how symmetrical the tumor is about its center.
fractal_dimension_mean: The mean of the fractal dimension, reflecting how detailed the tumor's surface is.

radius_se: The standard error of the radius measurement.
texture_se: The standard error of the texture measurement.
perimeter_se: The standard error of the perimeter measurement.
area_se: The standard error of the area measurement.
smoothness_se: The standard error of the smoothness measurement.
compactness_se: The standard error of the compactness measurement.
concavity_se: The standard error of the concavity measurement.
concave points_se: The standard error of the concave points measurement.
symmetry_se: The standard error of the symmetry measurement.
fractal_dimension_se: The standard error of the fractal dimension measurement.

radius_worst: The worst (largest) value of the radius measurements across all the samples.
texture_worst: The worst (largest) value of the texture measurements across all the samples.
perimeter_worst: The worst (largest) value of the perimeter measurements across all the samples.
area_worst: The worst (largest) value of the area measurements across all the samples.
smoothness_worst: The worst (largest) value of the smoothness measurements across all the samples.
compactness_worst: The worst (largest) value of the compactness measurements across all the samples.
concavity_worst: The worst (largest) value of the concavity measurements across all the samples.
concave points_worst: The worst (largest) value of the concave points measurements across all the samples.
symmetry_worst: The worst (largest) value of the symmetry measurements across all the samples.
fractal_dimension_worst: The worst (largest) value of the fractal dimension measurements across all the samples.

Basically numerous measurements have been made for each single tumor, and in this dataset they chose to show the average measurements, the worst (biggest) one and the squared error.