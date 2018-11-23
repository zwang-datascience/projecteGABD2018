virtualenv-2.7 -p python2 env
source venv/bin/activate

cd src
pip install -r requirements.txt
pip install cx_Oracle
pip install pymongo
cd ..
