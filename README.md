# Outliers Detection experiments:

## Contact

* [Oriol Ramos Terrades](oriol.ramos@uab.cat)
* Carles Sánchez Ramos

## Install

- Requires python 2.7
- Run script `./install.sh`

## Usage

* To insert the __Iris__ dataset from the __UCI__ repository into an __Oracle__ DBMS

```
    python src/insertData.py -c configImport.json -D oracle  -N UCI/Iris  -f <ROOT_PATH>/UCI/Iris.data.txt

```

* To insert the __Iris__ dataset from the __UCI__ repository into a __MongoDB__  DBMS

```
    python src/insertData.py -c configImport.json -D mongoDB  -N UCI/Iris  -f <ROOT_PATH>/UCI/Iris.data.txt

```
* To insert the __mirFlickr__ dataset  using the __AlexNet__ Network and __Visual__ Features  into an __Oracle__ DBMS

```
    python src/insertData.py -c configImport.json -D oracle -N MIRFLICKR/AlexNet -f <ROOT_PATH>/mirFlickr/features/AlexNet1_4000.Labelfeatures

```
* To insert the __mirFlickr__ dataset  using the __AlexNet__ Network and __Visual__ Features  into a __MongoDB__ DBMS

```
    python src/insertData.py -c configImport.json -D mongoDB -N MIRFLICKR/AlexNet -f <ROOT_PATH>/mirFlickr/features/AlexNet1_4000.Labelfeatures

```
* To see other parameters and options (--help):
```
 python src/insertData.py -h
```


## TODO

- [ ] Validar connexió a MongoDB.
