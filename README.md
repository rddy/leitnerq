Leitner Queue Network
=====================

A collection of Jupyter notebooks for analyzing and simulating the Leitner Queue Network. 
Read more about the model at http://siddharth.io/leitnerq.

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
Pkg.add("JuMP")
Pkg.add("Ipopt")
Pkg.add("Gadfly")
```

Navigate to the `nb` directory and start a Jupyter notebook server with

```
jupyter notebook
```

Download [this folder](https://www.dropbox.com/sh/epx7hzezh1ok6qe/AABkUeVSJXpmCjyxyag-uaHKa?dl=0)
and copy the `data`, `results`, and `figures` folders into the `nb` directory.

Questions and comments
----------------------

Please contact the author at `sgr45 [at] cornell [dot] edu` if you have questions or find bugs.
