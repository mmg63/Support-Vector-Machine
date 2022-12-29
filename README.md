



# Table of Contents


[Code Implementation	1](#_Toc117805782)

[Accuracy measure for each classifier	1](#_Toc117805783)

[Discussions and comparisons	1](#_Toc117805784)

[SVM-L (Linear kernel)	1](#_Toc117805785)

[SVM-P (polynomial kernel)	2](#_Toc117805786)

[SVM-R (Radial Basis Function kernel)	3](#_Toc117805787)

[Grid search for finding the best parameters	4](#_Toc117805788)

[Conclusion	4](#_Toc117805789)

[**Appendix**	5](#_Toc117805790)

[Performance measures for each dataset corresponding to the models	5](#_Toc117805791)

[The output of the program	6](#_Toc117805792)

[Decision boundary for each classifier	7](#_Toc117805793)

[ROC curve, AUC, and best parameters for SVM-R	8](#_Toc117805794)

[Classifiers Comparison	9](#_Toc117805795)

[**References**	10](#_Toc117805796)






# List of Figures

[Figure 1. SVM-L decision boundaries for different datasets.	2](#_Toc117805814)

[Figure 2. SVM-P decision boundaries for different datasets.	2](#_Toc117805815)

[Figure 3 SVM-R decision boundaries for different datasets.	3](#_Toc117805816)

[Figure 4. Screenshot from the output of the program	6](#_Toc117805817)

[Figure 5. Decision boundary for each classifier	7](#_Toc117805818)

[Figure 6 ROC curve, AUC for each dataset using RBF kernel.	8](#_Toc117805819)

[Figure 7 Classifier comparison	9](#_Toc117805820)



# List of Tables
[Table 1. Accuracy measure for each classifier	1](#_Toc117805876)

[Table 2. Best parameters for each dataset	4](#_Toc117805877)

[Table 3. performance measures on the Circles0.3 dataset	5](#_Toc117805878)

[Table 4. Performance measures on the halfkernel dataset.	5](#_Toc117805879)

[Table 5. Performance measures on the moons1 dataset	5](#_Toc117805880)

[Table 6. Performance measure on the Spiral1 dataset	5](#_Toc117805881)

[Table 7. Performance measure on the twoguassion33 dataset.	5](#_Toc117805882)

[Table 8. performance measure on the twogaussions42 dataset	6](#_Toc117805883)





# Code Implementation 
We created the program using Python 3.7. We used the scikit-learn library [1] to implement the machine learning methods. For this assignment, we import SVC from the SVM (support vector machine) class of the scikit library. The Dataset object stores the names of the datasets as tuples using pandas data frame. We created an ML-methods object to apply the methods to all the datasets. Furthermore, we split the data 80-20 for the training and testing the models. Moreover, we applied a min-max scaler on datasets to simplify the training procedure. We used the kernel parameter to define different kernels and created separate methods for them. We created individual programs (.ipynb) to apply grid-search on each dataset and used SVM-R to find its best parameters and accuracy. We used StratifiedKFold to perform 10-fold cross-validation and report the mean accuracy of all the performances and other desired metrics. Then, we used matplotlib [2] to plot the accuracy & decision boundary of each technique on each dataset. Finally, the source code, including all the generated plots, is available in a GitHub repository ([Link to the repository](https://github.com/mmg63/Support-Vector-Machine#_Toc117805882)).

# Accuracy measure for each classifier


|CLASSIFIER|CIRCLES0.3|HALFKERNEL|MOONS1|SPIRAL1|TWOGAUSSIANS33|TWOGAUSSIANS42|
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
|**SVM-L**|48|73|91|76|99|93|
|**SVM-P**|85|83|91|76|99|93|
|**SVM-R**|100|100|100|97|99|94|
*Table 1. Accuracy measure for each classifier*

# Discussions and comparisons
The SVM uses various kernels to draw a decision boundary by the farthest point of one class nearest to the other class. Although the kernel computes the distance between each point in the dataset, only the points nearest the decision boundary are needed most of the time. For instance, if we are classifying apples and oranges, then SVM would look at an orange that looks most like an apple and vice versa. In other words, it selects data points in cases which is very close to the boundary and uses that to construct its analysis. Therefore, SVM is different from most other ML algorithms. In this assignment, we implemented three different kernels for SVM, which will be discussed in what follows.

## SVM-L (Linear kernel)
In SVM linear kernel, the algorithm ensures the margin distance between points and the classifier line remains at maximum. Support vectors are the points nearest to the classifier line, which works as the core elements in SVM. i.e., Support vectors sustain the whole algorithm. Other points do not contribute to the algorithm. The main classifier line is known as the maximum margin hyperplane. 

- We found the linear kernel useless for the circles0.3 dataset, as the algorithm could not create any linear line separating the classes. In SVM, a parameter allows us to limit the points that mistakenly fall between the support vectors. However, in this case, no matter how we tune this parameter, many error points lead to finding no linear boundaries. Therefore, the accuracy remains under 50%, which means all the points are in one class.
- Somehow SVM-L got some fit on the halfkernel (73.75%) and spiral1 dataset (74.13%). By observing decision boundaries, we can see that although the accuracy is above average, we can not conclude the boundaries as a reasonable classification. Maybe when used with different kernels would provide better accuracy for these datasets. 
- SVM-Linear very well separated datasets like moons1 and twogaussian42. Nevertheless, we got the best results on twogaussian33 dataset. We were able to catch training and testing accuracy of 99% on the twogaussian33 dataset while using SVM-L.

Figure 1 shows the performance of the linear kernel regarding computing decision boundaries on all the datasets.

|<p>![](./Readme%20Images/Aspose.Words.3bcc7811-4ccd-490b-87ee-9755dc9f822a.001.png)![](./Readme%20Images/Aspose.Words.3bcc7811-4ccd-490b-87ee-9755dc9f822a.002.png)![](./Readme%20Images/Aspose.Words.3bcc7811-4ccd-490b-87ee-9755dc9f822a.003.png)![](./Readme%20Images/Aspose.Words.3bcc7811-4ccd-490b-87ee-9755dc9f822a.004.png)![](./Readme%20Images/Aspose.Words.3bcc7811-4ccd-490b-87ee-9755dc9f822a.005.png)![](./Readme%20Images/Aspose.Words.3bcc7811-4ccd-490b-87ee-9755dc9f822a.006.png)</p><p></p>|
| :-: |
*Figure 1. SVM-L decision boundaries for different datasets.*


## SVM-P (polynomial kernel)
This approach seeks a non-linear hyperplane as a boundary decision between the classes. Unlike the linear kernel, the model transforms the points into a higher dimension using the dot product of the pair points where each point has additional dimensions. These dimensions are derived from the points and their relationship with other points using the polynomial kernel. The kernel solves the non-line problem using a linear classifier in a higher dimension, which is known as the kernel trick. Based on the assumption, the polynomial degree is two. Thus, the SVM will find a curve with a degree of two to draw the decision boundary between classes. Therefore, the model performs well for Gaussian distribution points with a curve line to distinguish classes, as seen in the twogaussians33 dataset (99.25%). Furthermore, the accuracy decreases when the margin for support vectors contains noises that derive from overlapping points. For instance, the model achieved 90.88% accuracy on the twogaussians42 dataset. However, when a polynomial decision line of degree does not exist, the polynomial kernel cannot find the proper curve, and it needs more project the points in higher dimensions. Therefore, a non-linear line from the degree of cannot find the proper boundary for circles0.3 (88.5%), halfkernel (83.5%), moons1 (87.62%), and spiral (72.62%). Although the accuracy seems above expected, we can observe the poor decision boundary using the polynomial kernel in figure 2.

|<p>![](./Readme%20Images/Aspose.Words.3bcc7811-4ccd-490b-87ee-9755dc9f822a.007.png)![](./Readme%20Images/Aspose.Words.3bcc7811-4ccd-490b-87ee-9755dc9f822a.008.png)![](./Readme%20Images/Aspose.Words.3bcc7811-4ccd-490b-87ee-9755dc9f822a.009.png)![](./Readme%20Images/Aspose.Words.3bcc7811-4ccd-490b-87ee-9755dc9f822a.010.png)![](./Readme%20Images/Aspose.Words.3bcc7811-4ccd-490b-87ee-9755dc9f822a.011.png)![](./Readme%20Images/Aspose.Words.3bcc7811-4ccd-490b-87ee-9755dc9f822a.012.png)</p><p></p>|
| :-: |
*Figure 2. SVM-P decision boundaries for different datasets.*


## SVM-R (Radial Basis Function kernel)
When we want to use the support vector machine for non-linear datasets, it is useful to use the SVM-R for this reason. The Radial Basis Function is the most generalized form of kernelization and is one of the widest usages for measuring distance using the Gaussian Distribution. The RBF function for calculating the distance of two points $X_1$ and $X_2$ is:

```math
K(X_1,X_2)=exp\Big(-\frac{\|X_1-X_2\|^2}{2\sigma^2}\Big)
```

Where $\sigma$ is variance and the nominator equation is our L2-norm or the Euclidean distance. Now, Let  $L_{12}$ be the distance between the two points  $X_1$ and $X_2$; we can now represent $L_{12}$ as follows:

![Chart Description automatically generated with medium confidence](./Readme%20Images/Aspose.Words.3bcc7811-4ccd-490b-87ee-9755dc9f822a.013.jpeg)

![Chart Description automatically generated with low confidence](./Readme%20Images/Aspose.Words.3bcc7811-4ccd-490b-87ee-9755dc9f822a.014.png)When the points are near, the result of the formula is near one, and if  they are far, the result will be approached zero. It is essential to find the right value of 'σ' to decide which points should be considered similar, and this can be demonstrated on a case-by-case basis. When $\sigma=1$, $\sigma^2 = 1$, and the RBF kernel's mathematical equation are as follows:


```math
K(X1,X2)=exp\Big(-\frac{\|X_1-X_2\|^2}{2}\Big)
```

Turn to our datasets and models of use, The best way to use SVM in our datasets is SVM-R. As the plots are taken, we can see that the boundaries identified using SVM-R are more reliable.

|<p>![](./Readme%20Images/Aspose.Words.3bcc7811-4ccd-490b-87ee-9755dc9f822a.015.png)![](./Readme%20Images/Aspose.Words.3bcc7811-4ccd-490b-87ee-9755dc9f822a.016.png)![](./Readme%20Images/Aspose.Words.3bcc7811-4ccd-490b-87ee-9755dc9f822a.017.png)![](./Readme%20Images/Aspose.Words.3bcc7811-4ccd-490b-87ee-9755dc9f822a.018.png)![](./Readme%20Images/Aspose.Words.3bcc7811-4ccd-490b-87ee-9755dc9f822a.019.png)![](./Readme%20Images/Aspose.Words.3bcc7811-4ccd-490b-87ee-9755dc9f822a.020.png)</p><p></p>|
| :-: |
*Figure 3 SVM-R decision boundaries for different datasets.*


## Grid search for finding the best parameters
In SVM methods, choosing the right kernel is crucial, and finding the best parameters leads to great results. Therefore, one of the essential steps for achieving a great result is tuning the parameters. Grid search is an approach that trains various models using different parameters from a defined list of parameters, which leads to the best accuracy. Table 2 shows the best hyperparameters for each dataset. Not surprisingly, the best kernel for all the datasets is RBF. In addition, more information, such as the ROC curve and AUC, is available in the appendix. Finally, the list of parameters is shown is below sets.

```math
 C \in \{0.25, 0.5, 0.75, 1\}
 ```

```math
\gamma \in {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}
```

|DATASET / PARAM|$C$|$\gamma$|KERNEL|ACCURACY|
| :-: | :-: | :-: | :-: | :-: |
|**CIRCLES0.3**|0.25|0.4|RBF|100|
|**HALFKERNEL**|0.25|0.1|RBF|100|
|**MOONS1**|0.5|0.8|RBF|99.75|
|**SPIRAL1**|0.25|0.4|RBF|99.75|
|**TWOGAUSSIANS33**|0.25|0.6|RBF|99.50|
|**TWOGAUSSIANS42**|0.5|0.8|RBF|94|
*Table 2. Best parameters for each dataset*

# Conclusion
In conclusion, the SVM method performs very well if we choose the appropriate kernel considering the features of the datasets, such as the distribution and overlapping records. For the datasets that are linearly separatable, SVM-L can easily find a good margin between the classes, such as gaussian distributions with non-overlapping data points in the covariance matrix. However, when the overlapping points cause more errors through margins of support vector limits, there is no linear line that can draw the boundary. For instance, all the points in circle dataset have been categorized into one class using a linear kernel. The polynomial kernel is more effective for drawing the boundary, especially in datasets with guassian distribution. However, it is still unable to cluster non-gaussian datasets effectively since we have limited it to the degree of two. On the other hand, SVM-R is the most effective and well-known SVM technique we can often use in different types of datasets, whether they benefit from a gaussian distance or not. It can adequately cluster even circle03 and spiral 1 because it benefits from an infinite degree, which can finally draw a linear boundary in an endless dimension space. We can see that the decision boundaries identified using SVM-R are more reliable.


# **Appendix**

## Performance measures for each dataset corresponding to the models


||
| :-: |

|**circles0.3**|**Training Accuracy**|**PPV**|**NPV**|**Sensitivity**|**Specificity**|**Testing Accuracy**|
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
|**SVM-L**|48|0|0.48|0|1|0.48|
|**SVM-R**|100|1|1|1|1|1|
|**SVM-P**|88.5|0.78|1|1|0.7|0.85|

|***Table 3. performance measures on the Circles0.3 dataset***|
| :-: |


||
| :-: |

|**halfkernel** |**Training Accuracy**|**PPV**|**NPV**|**Sensitivity**|**Specificity**|**Testing Accuracy**|
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
|**SVM-L**|73.75|0.81|0.68|0.63|0.84|0.73|
|**SVM-R**|100|1|1|1|1|1|
|**SVM-P**|83.5|0.96|0.76|0.71|0.97|0.83|

|***Table 4. Performance measures on the halfkernel dataset.***|
| :-: |

||
| :-: |

|**moons1** |**Training Accuracy**|**PPV**|**NPV**|**Sensitivity**|**Specificity**|**Testing Accuracy**|
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
|**SVM-L**|87.25|0.92|0.89|0.88|0.93|0.91|
|**SVM-R**|99.75|1|1|1|1|1|
|**SVM-P**|87.62|0.9|0.91|0.91|0.9|0.91|

|***Table 5. Performance measures on the moons1 dataset***|
| :-: |


||
| :-: |

|**spiral1** |**Training Accuracy**|**PPV**|**NPV**|**Sensitivity**|**Specificity**|**Testing Accuracy**|
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
|**SVM-L**|74.13|0.76|0.76|0.71|0.8|0.76|
|**SVM-R**|98.75|0.95|0.99|0.99|0.95|0.97|
|**SVM-P**|72.62|0.79|0.74|0.66|0.85|0.76|

|***Table 6. Performance measure on the Spiral1 dataset***|
| :-: |


||
| :-: |

|**twogaussians33** |**Training Accuracy**|**PPV**|**NPV**|**Sensitivity**|**Specificity**|**Testing Accuracy**|
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
|**SVM-L**|99.13|1|0.98|0.98|1|0.99|
|**SVM-R**|99.25|1|0.99|0.99|1|0.99|
|**SVM-P**|99.25|1|0.99|0.99|1|0.99|

|***Table 7. Performance measure on the twoguassion33 dataset.***|
| :-: |


||
| :-: |

|**twogaussians42** |**Training Accuracy**|**PPV**|**NPV**|**Sensitivity**|**Specificity**|**Testing Accuracy**|
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
|**SVM-L**|89.88|0.99|0.88|0.85|0.99|0.93|
|**SVM-R**|93.25|1|0.9|0.87|1|0.94|
|**SVM-P**|90.88|1|0.88|0.85|1|0.93|

|***Table 8. performance measure on the twogaussions42 dataset***|
| :-: |

## The output of the program


|![](./Readme%20Images/Aspose.Words.3bcc7811-4ccd-490b-87ee-9755dc9f822a.021.jpeg)![](./Readme%20Images/Aspose.Words.3bcc7811-4ccd-490b-87ee-9755dc9f822a.022.jpeg)|
| :-: |
*Figure 4. Screenshot from the output of the program*


## ![](./Readme%20Images/Aspose.Words.3bcc7811-4ccd-490b-87ee-9755dc9f822a.023.png)Decision boundary for each classifier

*Figure 5. Decision boundary for each classifier*
## ROC curve, AUC, and best parameters for SVM-R


|![A picture containing graphical user interface

Description automatically generated](./Readme%20Images/Aspose.Words.3bcc7811-4ccd-490b-87ee-9755dc9f822a.024.png)![A picture containing graphical user interface

Description automatically generated](./Readme%20Images/Aspose.Words.3bcc7811-4ccd-490b-87ee-9755dc9f822a.025.png)![Graphical user interface

Description automatically generated with medium confidence](./Readme%20Images/Aspose.Words.3bcc7811-4ccd-490b-87ee-9755dc9f822a.026.png)![Shape

Description automatically generated with low confidence](./Readme%20Images/Aspose.Words.3bcc7811-4ccd-490b-87ee-9755dc9f822a.027.png)![Shape

Description automatically generated with low confidence](./Readme%20Images/Aspose.Words.3bcc7811-4ccd-490b-87ee-9755dc9f822a.028.png)![Shape

Description automatically generated with low confidence](./Readme%20Images/Aspose.Words.3bcc7811-4ccd-490b-87ee-9755dc9f822a.029.png)|
| :- |
*Figure 6 ROC curve, AUC for each dataset using RBF kernel.*



## Classifiers Comparison


|![](./Readme%20Images/Aspose.Words.3bcc7811-4ccd-490b-87ee-9755dc9f822a.030.png)![](./Readme%20Images/Aspose.Words.3bcc7811-4ccd-490b-87ee-9755dc9f822a.031.png)![](./Readme%20Images/Aspose.Words.3bcc7811-4ccd-490b-87ee-9755dc9f822a.032.png)![](./Readme%20Images/Aspose.Words.3bcc7811-4ccd-490b-87ee-9755dc9f822a.033.png)![](./Readme%20Images/Aspose.Words.3bcc7811-4ccd-490b-87ee-9755dc9f822a.034.png)![](./Readme%20Images/Aspose.Words.3bcc7811-4ccd-490b-87ee-9755dc9f822a.035.png)|
| :- |

*Figure 7 Classifier comparison* 


# **References**

[1]	D. K. Barupal and O. Fiehn, “Scikit-learn: Machine Learning in Python,” *J. ofMachine Learn. Res.*, vol. 12, pp. 2825–2830, 2011, doi: 10.1289/EHP4713.

[2]	J. D. Hunter, “Matplotlib: A 2D Graphics Environment,” *Comput. Sci. Eng.*, vol. 9, no. 3, pp. 90–95, 2007, doi: 10.1109/MCSE.2007.55.3


