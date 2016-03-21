# C++ API

## Run a simple (yet advanced) example
TODO: this is now part of dali-examples

Let's run a simple example. We will use data from [Paul Graham's blog](http://paulgraham.com) to train a language model. This way we can generate random pieces of startup wisdom at will! After about 5-10 minutes of training time you should see it generate sentences that sort of make sense. To do this go to `build` and call:

```bash
examples/language_model --flagfile ../flags/language_model_simple.flags
```

* A more extensive example for training a language model can be found under: `examples/language_model.cpp`.
* For a more in-depth description of usage see the [character model tutorial](docs/CharacterModel.md)
* For a funny example where you teach stacked LSTMs about multiplication, substraction, and addition [check this out](docs/Arithmetic.md).


## Utils

In the utilities namespace you will find several tools to make data processing and saving easier.

To create folders similar to how [os.makedirs](https://docs.python.org/2/library/os.html#os.makedirs) works in Python, you can do:

```cpp
utils::makedirs("folder/subfolder/");
```

Random integer between 0 and 2 (included):

```cpp
utils::randint(0, 2);
```

Check whether a file is gzipped:

```cpp
utils::is_gzip("folder/suspicious.gz");
```

Sort the arguments of a list [np.argsort](http://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html) style:

```cpp
auto sorted_lengths = utils::argsort(lengths);
```
