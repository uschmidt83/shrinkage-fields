The code in this package implements image deconvolution and denoising as described in the paper:

  Uwe Schmidt and Stefan Roth.
  Shrinkage Fields for Effective Image Restoration.
  IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Columbus, Ohio, June 2014.

Please cite the paper if you are using this code in your research.
Please see the file LICENSE.txt for the license governing this code.

  Version:       1.1 (19/11/2014), see CHANGELOG.txt
  Contact:       Uwe Schmidt <uwe.schmidt@gris.tu-darmstadt.de>
  Project page:  http://www.gris.tu-darmstadt.de/research/visinf/software/index.en.htm
  GitHub:        https://github.com/uschmidt83/shrinkage-fields


Overview
--------
The function CSF_DEMO demonstrates denoising and deconvolution with the learned models from the paper, which can all be found in the folder "models" (the respective sub-folders indicate the relation to the paper). The paper and supplemental material are provided in the folder "paper".
Learning is implemented in CSF_TRAIN, which relies on functions from the folder "+train", and uses example data from folder "data".
Shrinkage functions are represented by RBFMIX objects, see folder "@rbfmix" for details.
The folder "+misc" contains miscellaneous helper functions.

An overview of the functions in a particular folder can be displayed by typing "help <folder/package>" at the MATLAB prompt (e.g. "help train").


Dependencies
------------
This code depends on MATLAB with the image processing toolbox and has been tested with R2010a and newer versions.
It further requires a gradient-based optimization algorithm for learning with CSF_TRAIN (see file "+train/get_minimizer.m").
The function CSF_PREDICT will make use of a supported GPU, if available.
For improved runtime performance, it is recommended to compile the MEX-plugin "@rbfmix/private/lut_eval.c"; compiled binaries are already provided for 64-bit Linux and Windows systems (with MATLAB R2012a).


Contact
-------
If you have questions, problems with the code, or found a bug, please let us know.
Contact Uwe Schmidt at uwe.schmidt@gris.tu-darmstadt.de or mail@uweschmidt.org.
