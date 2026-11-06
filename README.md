# Oblig2_simulering_visualisering

## Setup
This project uses **Python 3.12**
```bash
#Check which version you have in bash. If it fails install python 3.12 from python.org
python3.12 --version   

#create and activate virtual enviroment (once)
python3.12 -m venv .venv
source .venv/bin/activate 

#install the required libraries
pip install -r requirements.txt
```
## How to run
From the root of the project run:
```bash
python src/del_1.py src/del_2.py 
panel serve app.py --show --dev
```
