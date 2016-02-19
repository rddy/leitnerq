Leitner Queue Network
=====================

A collection of Jupyter notebooks for analyzing and simulating the Leitner Queue Network. 
Read more about the model at http://siddharth.io/leitnerq.

Usage
-----

Install the notebook dependencies with 

```
pip install -r requirements.txt
```

Install IJulia using the instructions [here](https://github.com/JuliaLang/IJulia.jl). 
Navigate to the notebook directory and start a Jupyter server with

```
cd nb
jupyter notebook
```

`lqn_properties` is for analyzing the properties of the Leitner Queue Network. `lqn_simulations` 
is for simulating the Leitner Queue Network. `mturk_experiments` is for doing exploratory analysis
of the log data collected from the Mechanical Turk experiments. `mnemosyne_data` is for doing
exploratory analysis of the Mnemosyne log data. `mnemosyne_evaluation` is for evaluating memory
models on the Mnemosyne log data. 

To download the data and results used in the project manuscript, download [this folder](https://www.dropbox.com/sh/epx7hzezh1ok6qe/AABkUeVSJXpmCjyxyag-uaHKa?dl=0)
and copy the `data` and `results` folders into the `nb` directory.

Questions and comments
----------------------

Please contact the author at `sgr45 [at] cornell [dot] edu` if you have questions or find bugs.
