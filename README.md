![alt text](https://github.com/kdd-anonymous/rankfromsets/raw/master/arxiv_user_embeddings_tsne.png)

# RankFromSets

This code accompanies the RankFromSets SDSS submission. 

To view the above visualization in a browser, please download [this HTML file](https://github.com/altosaar/rankfromsets/raw/master/(Download%20and%20open%20in%20a%20browser!)%20rankfromsets-65k-arXiv-user-embeddings-t-SNE.html).

## Environment

These experiments were conducted on a red hat linux cluster with Nvidia P100
GPUs.

Python environment, using the Anaconda python package manager:
```
conda env create -f environment.yml
```

## Data format
`{train, valid, test}.tsv` files are observations of user, item interactions.

The `item_attributes_csr.npz` is a [compressed sparse row](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html) format matrix of shape `(n_items, n_attributes)`. For example, if the data is documents in a bag of words format, each row is a document and the attributes are the words.

## Synthetic Example

We omit raw arXiv data and food tracking data as it is private user data.

This follows the reproducibility supplement example of a square kernel.

Generate data to `/tmp/dat/simulation_%d` where %d is a number from 1 to 30 replications:

```
export DAT=/tmp
```

```
python build_simulation_dataset.py
```

Launch the best-performing parameter settings with the SLURM manager for the inner product, deep, and residual regression functions:

```
PYTHONPATH=. python experiment/arxiv/grid.py
```

## Hyperparameters

We find that large batch sizes significantly improve performance. See `config.yml` for the best-performing hyperparameters.
