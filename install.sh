virtualenv -p python2 env
source env/bin/activate

cd src
pip install -r requirements.txt
pip install cx_Oracle
pip install pymongo
cd ..
