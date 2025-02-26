


The files and their functions are described here


## Basic functions `ECCO_helper.py`

It contains the methods to get ECCO data filenames, load LLC grid, and most importantly, compute the tendency term (`computeTendency`).


## Derive mixed layer related terms

Run `postprocess_ECCO_all.sh`, it runs `postprocess_ECCO.py` which loads the library `postprocess_ECCO_tools.py`.


- `postprocess_ECCO_all.sh` is the shell to postprocess data with various parameters. 
- `postprocess_ECCO.py` is the parallel program that execute `postprocess_ECCO_tools.py` to compute the mixed layer integrated terms.
- `postprocess_ECCO_tools.py` compute the mixed layer integrated terms.

