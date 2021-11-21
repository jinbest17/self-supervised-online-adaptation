How to run

run unsupervised baselines on gas:
python3 run.py gas ot
python3 run.py gas lssvm

run our model on gas:
python3 run.py gas SOA

run from saved weights:
python3 run.py gas SOA from_saved

run our model on MNIST:
python3 run.py mnist SOA noise
python3 run.py mnist SOA rotate
python3 run.py mnist SOA permutation


run our model on MNIST from saved weights:
python3 run.py mnist SOA noise from_saved
python3 run.py mnist SOA rotate from_saved
python3 run.py mnist SOA permutation from_saved

DOESN'T WORK RIGHTNOW
ot on mnist from saved weights:
python3 run.py mnist ot noise from_saved 
python3 run.py mnist ot rotate from_saved
python3 run.py mnist ot permutation from_saved