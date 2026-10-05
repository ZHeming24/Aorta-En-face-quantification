# Aorta-En-face-quantification
Oil Red O Aorta QC & Quantification is an interactive macOS desktop application designed for semi-automated analysis of en face aortic Oil Red O staining. The program combines manual anatomical guidance with automated image segmentation, allowing the user to define the relevant aortic region, optimize plaque detection, visually verify the results, and export quantitative measurements in physical units.

The workflow is designed to reduce segmentation errors caused by rulers, needles, background reflections, uneven illumination, and anatomical variation between samples.

Main Features

* Image preprocessing
    * Adjustable brightness
    * Adjustable contrast
    * Adjustable gamma
    * All image adjustments are applied consistently to downstream segmentation and QC output
* Image cropping
    * Interactive crop selection before analysis
    * Allows removal of irrelevant background regions
    * Crop can be reset at any time
* Zoom and navigation
    * Zoom in / zoom out controls
    * Mouse wheel or trackpad zooming
    * Image panning for detailed inspection
* Physical calibration using a ruler
    * The user selects two ruler marks with a known physical distance
    * The program calculates the image scale in mm/pixel
    * All length and area measurements are subsequently reported in mm and mm², rather than pixels
* Interactive rough aorta selection
    * The user defines the approximate aortic region by holding the Command key and moving the pointer around the specimen
    * No trackpad click-and-drag is required
    * The rough region can be redrawn if necessary
* Automatic aorta mask refinement
    * The program refines the manually defined rough region using image color and morphology
    * Automated segmentation is restricted to the user-defined anatomical area, reducing false detection of background objects
* Interactive plaque threshold adjustment
    * Oil Red O-positive plaque is detected using a red-color intensity score
    * A slider allows the plaque threshold to be adjusted for each sample
    * Changes are displayed in real time
* Visual QC overlay
    * Blue: aorta boundary
    * Yellow: detected plaque
    * Cyan: centerline used for aortic length measurement
    * Green: ruler calibration line
    * Red: manually drawn rough aorta region
* QC confirmation
    * Each sample must be manually reviewed and confirmed before inclusion in the final analysis
    * Unconfirmed samples are excluded from batch export

Quantitative Measurements

For each QC-approved sample, the program calculates:

* Aorta area (mm²)
* Plaque area (mm²)
* Plaque area percentage (%)
* Principal aortic centerline length (mm)
* Diagnostic total skeleton length (mm)
* Image scale (mm/pixel)
* Plaque threshold
* Image adjustment parameters, including brightness, contrast, and gamma

Aortic Length Measurement

The primary aortic length measurement is based on the principal centerline, rather than the sum of all skeleton branches.

The analysis follows these steps:

1. The accepted aorta mask is smoothed to reduce small segmentation irregularities.
2. The mask is skeletonized to generate a one-pixel-wide centerline network.
3. Skeleton endpoints are identified.
4. Geodesic distances between endpoints are calculated along the skeleton.
5. The longest continuous endpoint-to-endpoint path is selected.
6. This path is converted from pixels to millimeters using the ruler calibration.

The exact path used for length measurement is displayed as a cyan line in the QC image, allowing direct visual verification.

This approach is particularly useful for Y-shaped aortic preparations because it avoids artificially inflating the length by summing all branches and small skeleton artifacts.

QC Output

For each confirmed specimen, the program generates:

* Individual QC overlay image
* Aorta mask
* Plaque mask
* Centerline mask
* Combined QC image
    * Left panel: adjusted and cropped original image
    * Right panel: QC overlay

Data Export

Batch analysis exports the results as:

* CSV
* Excel
* QC overlay images
* Combined QC images
* Binary segmentation masks

The application is intended for semi-automated quantitative analysis of Oil Red O-stained aortas with mandatory visual quality control, providing a reproducible workflow while retaining user control over anatomical selection and plaque thresholding.
