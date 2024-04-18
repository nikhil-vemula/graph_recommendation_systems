# Download Data Set

## Create virtual enviroment

```
python -m venv graph_rs
source graph_rs/bin/activate
pip install -r requirements.txt
```

```
curl -L https://files.grouplens.org/datasets/movielens/ml-latest-small.zip | bsdtar -xvf - -C data
```