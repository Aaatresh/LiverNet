# LiverNet

## Description
This repository outlines the implementation of the LiverNet model in Python using the TensorFlow deep learning framework. This model can perform effective and efficient cancer grading given H&E stained histopathology images of liver tissue.

Link to publication: [LiverNet: efficient and robust deep learning model for automatic
diagnosis of sub-types of liver hepatocellular carcinoma cancer from
H&E stained liver histopathology images](https://link.springer.com/article/10.1007/s11548-021-02410-4).

## Getting Started

### Dependencies
* TensorFlow = 2.3
* OpenCV = 4.1
* Numpy = 1.19
* Scikit-learn = 0.22
* Matplotlib = 3.3

### Installing
```
  git clone https://github.com/Aaatresh/LiverNet
```

### Executing code

Running the program:
* Use Jupyter Notebook or google colab to open and run each cell of the notebooks present in the 'code' folder. Instructions in the notebook guide the user through training and testing different networks, including LiverNet.
* If using jupyter notebook:
```
# [in BASH]
cd LiverNet/code 
jupyter notebook
```
* If using colab, upload this notebook to the drive.
* Execute cells by following the instructions.

### Model weights
The fold-wise trained model weights are available in the 'weights' folder for both KMC and TCGA datasets.


## Dataset
To request access to the datasets used in our work, please contact [Prof. Shyam Lal](https://ece.nitk.ac.in/faculty/shyam-lal).

## Authors

Contributors' names and contact info:
* Anirudh Aatresh ([anirudhashokaatresh@gmail.com](mailto:anirudhashokaatresh@gmail.com))
* Kumar Alabhya ([kumar.alabhya@gmail.com](mailto:kumar.alabhya@gmail.com))
* Shyam Lal ([shyam.mtec@gmail.com](mailto:shyam.mtec@gmail.com))
* Jyoti Kini ([kinijyoti@gmail.com](mailto:kinijyoti@gmail.com))
* PU Prakash Saxena ([pu_saxena83@yahoo.co.in](mailto:pu_saxena83@yahoo.co.in))

## Version History
* 1.0
    * Initial Release

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.

## Acknowledgments
* The authors thank the department of 
pathology, Kasturba Medical College (KMC) Mangalore, Manipal
Academy of Higher Education (MAHE), Manipal, Karnataka, India,
for providing the liver cancer histopathology image dataset.
* The authors thank the Science Engineering and Research Board, Department of Science and Technology, 
Govt. of India, for the grant provided to fund this research. 
