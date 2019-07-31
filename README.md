# gridlod-on-perturbations

```
# This file is part of the project for "Numerical Upscaling of perturbed diffsuion problems":
#   https://github.com/gridlod-community/gridlod-on-perturbations.git
# Copyright holder: Tim Keil, Fredrik Hellman 
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
```

This repository contains the application code for python module 'gridlod'. This module has been developed by Fredrik Hellman and Tim Keil on https://github.com/fredrikhellman/gridlod. Our repository provides further classes, that work as an extension of 'gridlod' and enable an efficient construction of two dimensional coefficients. 

## Setup

First you need to open a terminal at your favorite folder and clone the git repository that contains the 'gridlod' module. Furthermore, you switch to the compatible branch on which the extensions are established.

```
git clone https://github.com/gridlod.git
cd gridlod
git checkout master
cd ..
```

Now, add our repository and checkout the correct branch with
``` 
git clone https://github.com/gridlod-community/gridlod-on-perturbations.git
```

In order to connect 'gridlod' with our new files, you need to work with a virtual environment. First, you construct this environment and then you activate it.

```
virtualenv -p python2 venv2
. venv2/bin/activate
```

In this 'virtualenv', you have to install the required packages for python and everything that remains to run 'gridlod' and our files. 

```
pip install numpy scipy cython 
pip install matplotlib notebook ipython ipdb ipyparallel
```

If you are working on a mac, you need to use a slightly hacked version of scikit-sparse and do

```
brew install suite-sparse    <-- with deactivated virtualenv 
git clone https://github.com/TiKeil/scikit-sparse.git
cd scikit-sparse/
git checkout 0.2-homebrew
python setup.py install
pip install scikit-sparse
```

On Linux just install scikit-sparse (suite-sparse is required).

```
pip install scikit-sparse
```

Now, link gridlod to our folder that enables to use it as a module (sudo required).

```
echo $PWD/gridlod/ > $(python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())')/gridlod.pth
echo $PWD/gridlod-on-perturbations/ > $(python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())')/gridlod-on-perturbations.pth
```

Once this setting is done you can start working with our code.
