Leitner Queue Network
=====================

A collection of Jupyter notebooks for analyzing and simulating the Leitner Queue Network.
Read more about the model at https://people.eecs.berkeley.edu/~reddy/leitnerq

Usage
-----

Install Python dependencies with the [pip](https://pip.pypa.io/en/stable/installing/) package
manager using

```
pip install -r requirements.txt
```

Install the `lentil` package using the instructions [here](https://github.com/rddy/lentil). Install
[Julia](http://julialang.org/downloads/platform.html) dependencies with

```
Pkg.update()
Pkg.add("IJulia")
Pkg.add("JuMP")
Pkg.add("Ipopt")
Pkg.add("Gadfly")
```

To ensure that all plotting functions work, install [latex](https://www.latex-project.org/). Navigate to
the `nb` directory and start a Jupyter notebook server with

```
jupyter notebook
```

Download [this folder](https://www.dropbox.com/sh/epx7hzezh1ok6qe/AABkUeVSJXpmCjyxyag-uaHKa?dl=0)
and copy the `data`, `results`, and `figures` folders into the `nb` directory.

Questions and comments
----------------------

Please contact the author at `sgr45 [at] cornell [dot] edu` if you have questions or find bugs.

Citation
--------
If you find this software useful in your work, we kindly request that you cite the following [paper](http://arxiv.org/abs/1602.07032):

```
@InProceedings{Reddy/etal/16d,
  title={Unbounded Human Learning: Optimal Scheduling for Spaced Repetition},
  author={Reddy, Siddharth and Labutov, Igor and Banerjee, Siddhartha and Joachims, Thorsten},
  booktitle={Arxiv 1602.07032},
  year={2016},
  url={http://arxiv.org/abs/1602.07032}
}
```
