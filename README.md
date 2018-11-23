# Outliers Detection experiments:

## Contact

* [Oriol Ramos Terrades](oriol.ramos@uab.cat)
* Carles Sánchez Ramos

## Install

- Requires python 2.7
- Run script `./install.sh`

## Usage


Edit the config file: __configImport.json__ and modify the connection information:
```
{
"dbms" :{"oracle": {"username" : "<ORACLE_USER>",
			"sid": "ee.oracle.docker",
			"servers":{ "0": { "ssh" : {"username": "student",
						"hostname": "dcccluster.uab.es",
 						"port": "<YOUR_SSH_IP>"},
					"hostname" : "oracle-1.grup<YOUR_GROUP_NUMBER>.gabd",
					"port" : "1521"
					}
			}
		},
		"mongodb": {"username" : "<MongoDB_USER>",
			"db": "<YOUR_DB>",
			"servers":{ "0": { "ssh" : {"username": "student",
						"hostname": "dcccluster.uab.es",
 						"port": "<YOUR_SSH_IP>"},
					"hostname" : "main.grup<YOUR_GROUP_NUMBER>.gabd",
					"port" : "27017"
					}
			}}
	}
   }
```
Then run the scripts as the following examples:

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



