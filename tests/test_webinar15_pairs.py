"""Webinar 15: explicit correctness and edge-case checks."""
from pathlib import Path
import importlib.util
spec=importlib.util.spec_from_file_location('pairs',Path(__file__).resolve().parents[1]/'experiments'/'webinar15_pairs.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
find_pairs=module.find_pairs

def test_standard():assert find_pairs([1,2,3,4,5],6)==[(1,5),(2,4)]
def test_empty():assert find_pairs([],6)==[]
def test_single():assert find_pairs([3],6)==[]
def test_duplicate_pair_requires_two_occurrences():
    assert find_pairs([3,3],6)==[(3,3)]
    assert find_pairs([3],6)==[]
def test_negative_and_zero():assert find_pairs([-3,0,3,6],3)==[(-3,6),(0,3)]
def test_deduplicated_and_unmodified():
    data=[2,2,4,4];assert find_pairs(data,6)==[(2,4)];assert data==[2,2,4,4]
