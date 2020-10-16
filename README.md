# Sparse Low-Rank Decomposition for Graph Data Sets

This repository contains the code for the paper of the same name. The main function is to decompose graphs in a data set into a common sparse, low-rank template and sparse low-rank deviations from it.

## Requirements

The code was tested on Python 3.6, 3.7 and 3.8, and requires a version greater or equal to 3.4. In order to run the code, you need the Python packages NumPy, NetworkX, scikit-optimize and tqdm. They can be installed with `conda`:

```
conda create -n splr -c conda-forge --file requirements.txt
conda activate splr
pip install pybind11
pip install cpalgorithm
```

The running times reported were measured on a 2,9 GHz Intel Core i5 double core processor.

## Experiments on simulated data

In the paper, we present several experiments on noisy simulated matrices. We apply Sparse Low-Rank decomposition to estimate an underlying sparse low-rank template and the sparse low-rank deviations from it. The first case is a simple visual example. For the second case, we perform further numerical analysis, and compute the estimation error.

- In order to reproduce the results on the visual example of section 5.1 (duration: under a minute), run the following command. The parameters are pre-selected to gain time.
  
  ```
  python simple_example_fast.py
  ```

- For the same experiment with full parameter selection (duration: 18'00''), run
  
  ```
  python simple_example.py
  ```

- For the results of section 5.2 for random matrices with **unconstrained signs** (duration: 2h45'), run
  
  ```
  python estimation_unconstrained.py
  ```

- For the results of section 5.2 for random matrices with **symmetric positive coefficients** (duration: 3h15'), run
  
  ```
  python estimation_positive.py
  ```

The results will be stored in the `output` folder, and the figures in the `figures` folder. All the parameters for the data simulation and the optimization can be modified in the script headers.

The code includes an `expected_output` folder, which contains the result of the three commands above. Running them will produce the same result in the `output` folder.

#### Results table

For the experiments in `estimation_unconstrained.py` and `estimation_positive.py`, we evaluate the reconstruction of the template and the deviations by computing their relative Root Mean Square Errors (RMSE). The SPLRD model is compared with the naive decomposition `A = mean + (A-mean)`. We also compare with a similar decomposition where a sparse low-rank denoising (from Richard et al., 2012 [1]) is applied first to the mean to get a template estimate `TM`. The difference `A-TM` is then denoised similarly to obtain a deviation `VM`.

In the case of matrices with positive symmetric coefficients, we also compute the relative RMSE for several graph features: the average weighted degree d, the average shortest path length L with decreasing edge cost and the weighted clustering coefficient C.

In this table, each case shows the average relative RMSE for the given quantity, as well as a standard deviation of the error over the 5 test data sets.

|                    |     | SPLRD     | (TM, VM)  | (M,A-M)   |
| ------------------ | --- | --------- | --------- | --------- |
| Unconstrained sign | T   | .04 (.01) | .42 (.08) | .55 (.10) |
|                    | V   | .33 (.02) | .49 (.02) | .91 (.05) |
| Positive symmetric | T   | .07 (.02) | .55 (.22) | .64 (.27) |
|                    | V   | .30 (.01) | .55 (.02) | .77 (.03) |
| Graph features     | d   | .01 (.01) | .09 (.04) | .27 (.09) |
|                    | L   | .15 (.07) | .29 (.08) | .49 (.07) |
|                    | C   | .03 (.01) | .35 (.03) | .09 (.02) |

## Experiments on a real data set

In order to run the experiments on the US airline network of section 5.1 (duration: 2'30''), run the command:

```
python example_airline.py
```

The result will be stored in the `figures` folder.

## Run the algorithm on other data sets

In order to perform a sparse low-rank decomposition on a numpy array of adjacency matrices `samples` with shape `(n_samples,n,n)`, use:

```python
import src.douglas_rachford as dr

model = dr.SparseLowRank(lambd=..., rho=..., mu=..., nu=..., theta=0.9, tau=0.1, niter=200)
model.fit(samples)
```

The estimated template can be found in `model_L1.T`, and the list of deviations (with shape `(n_samples,n,n)`) in `model_L1.Vs`. These estimates are the low-rank version of the solution (denoted T_\* and V_\* in the paper). For the sparse version (T_1 and V_1), use `model_L1.S` and `model_L1.Ws` instead.

## Data source

The airplane traffic record can be found in the material of Williams et Musolesi [2]. The file `flightsFAA_stnet.csv` contains the list of flights for each hour over 10 days. The file `flightsFAA_coords.csv` contains the geographical coordinates of the airports used in the previous file, which allows to identify an airport if necessary. The data set is available under CC-By Attribution 4.0 International licence, which allows to copy, share and work on the data freely as long as the licence terms are respected.

## References

[1] Richard, E., Savalle, P.-A., & Vayatis, N. (2012). Estimation of Simultaneously Sparse and low-rank Matrices. In ICML 2012. http://arxiv.org/abs/1206.6474. Accessed 21 February 2020

[2] Williams, M. J., & Musolesi, M. (2016). Spatio-temporal networks: reachability, centrality and robustness. Royal Society Open Science, 3(6), 160196. https://doi.org/10.1098/rsos.160196

[3] Waschke, L., Alavash, M., Obleser, J., & Erb, J. (2018). AUDADAPT. https://doi.org/10.17605/OSF.IO/28R57
