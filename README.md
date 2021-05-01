# Dissertation

All the code and LaTeX foo needed to reproduce my thesis with a single call to `make`

## Requirements

Use `pyenv` or similar tools to install Python 3.7.10.
Then install `poetry`.
Install `graphviz` first. On MacOS I use `brew`.

```
brew install graphviz
````
The Latex code was slightly adapted from the original published version in order to work with TexLive / MacTex 2021.

Now clone the repository and perform `poetry install` to install all dependencies from the lock file.

All input data has to go into the `./data` directory. Links to the data can hopefully be provided in a future update. For now do not hesitate to contact me.


## Build the Thesis


Go  into the projects root directory and call 

```
MPLBACKEND=pgf poetry run make

```

Now wait for anywhere between 1 and 2 hours depending on your CPU speed. The final result will be written to `build/thesis.pdf`

