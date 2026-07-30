import random
from pathlib import Path

def test_random():
    for i in range(10):
        print(int(random.random()*100))

def test_path():
    cur_file_abs_path=Path(__file__).resolve()
    print(cur_file_abs_path)
    #get the last 2 level parent abs path
    parents_file_abs_path = Path(__file__).resolve().parents[2]
    print(parents_file_abs_path)

test_path()
