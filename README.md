# gradient-based-inference
Code for "Lee, J. Y., Mehta, S. V., Wick, M., Tristan, J. B., & Carbonell, J. (2019, July). [Gradient-based inference for networks with output constraints](https://www.aaai.org/ojs/index.php/AAAI/article/view/4316). In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 33, pp. 4147-4154)."

## Requirements
Python 3.6, PyTorch 0.4.1, AllenNLP v0.4.1

### Setting up a virtual environment

[Conda](https://conda.io/) can be used to set up a virtual environment
with Python 3.6 in which you can
sandbox dependencies required for our implementation:

1.  [Download and install Conda](https://conda.io/docs/download.html).

2.  Create a Conda environment with Python 3.6

    ```
    conda create -n gbi python=3.6
    ```

3.  Activate the Conda environment.  (You will need to activate the Conda environment in each terminal in which you want to run our implementation).

    ```
    source activate gbi
    ```

### Setting up our environment

1. Visit http://pytorch.org/ and install the PyTorch 0.4.1 package for your system.

2.  Clone our repo:

    ```
    git clone git@github.com:sanketvmehta/gradient-based-inference.git
    ```

#### Installing AllenNLP from source

1.  Clone ``allennlp`` with git submodule
    ```
    git submodule update --init
    ```

2. Checkout ``allennlp`` to ``v0.4.1``
    ```
    git checkout 31f4f60
    ```

3.  Change your directory to ``allennlp`` submodule present under the parent repo directory:

    ```
    cd gradient-based-inference/allennlp
    ```

4. Install the necessary requirement by running 

   ```
   INSTALL_TEST_REQUIREMENTS=true scripts/install_requirements.sh
   ```

5. Once the requirements have been installed, run:

   ```
   pip install --editable .
   ```

6. Test AllenNLP installation by running:

   ```
   ./scripts/verify.py
   ``` 
That's it! You're now ready to reproduce our results.

### Citing

If you use our code in your research, please cite: [Gradient-based inference for networks with output constraints](https://www.aaai.org/ojs/index.php/AAAI/article/view/4316).  

   ```
   @inproceedings{lee2019gradient,
  title={Gradient-based inference for networks with output constraints},
  author={Lee, Jay Yoon and Mehta, Sanket Vaibhav and Wick, Michael and Tristan, Jean-Baptiste and Carbonell, Jaime},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={33},
  pages={4147--4154},
  year={2019}
}
   ```