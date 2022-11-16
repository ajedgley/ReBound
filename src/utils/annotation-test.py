import unittest
import json
from deepdiff import DeepDiff # pip install deepdiff

input_path = '../../../a060c4c1-b9fc-39c1-9d30-d93a124c9066'
output_path = '../../../a060c4c1-b9fc-39c1-9d30-d93a124c9066-copy'
frame = '0'

# how to run tests: scuffed edition
# python annotation-test.py ClassName 
# ex: python annotation-test.py TestNoChange

# tests that if no change is made, the general format files are the same
class TestNoChange(unittest.TestCase):
    def test_no_change(self):
        in_str = str(input_path + "/bounding/" + frame + "/boxes.json")
        out_str = str(output_path + "/bounding/" + frame + "/boxes.json")

        with open(in_str) as in_file:
            json_dict1 = json.load(in_file)
        with open(out_str) as out_file:
            json_dict2 = json.load(out_file)

        diff = DeepDiff(json_dict1, json_dict2, ignore_order=True)
        self.assertTrue((diff == {}), diff)

# tests that if an insertion is made, the new general format file has the insertion
class TestInsert(unittest.TestCase):
    def test_insert(self):
        in_str = str(input_path + "/bounding/" + frame + "/boxes.json")
        out_str = str(output_path + "/bounding/" + frame + "/boxes.json")

        with open(in_str) as in_file:
            json_dict1 = json.load(in_file)
        with open(out_str) as out_file:
            json_dict2 = json.load(out_file)

        diff = DeepDiff(json_dict1, json_dict2, ignore_order=True)
        self.assertTrue(list(diff.keys()) == ['iterable_item_added'] and len(diff) == 1, diff)

# tests that if an deletion is made, the new general format file no longer has the box
class TestDelete(unittest.TestCase):
    def test_delete(self):
        in_str = str(input_path + "/bounding/" + frame + "/boxes.json")
        out_str = str(output_path + "/bounding/" + frame + "/boxes.json")

        with open(in_str) as in_file:
            json_dict1 = json.load(in_file)
        with open(out_str) as out_file:
            json_dict2 = json.load(out_file)

        diff = DeepDiff(json_dict1, json_dict2, ignore_order=True)
        self.assertTrue(list(diff.keys()) == ['iterable_item_removed'] and len(diff) == 1, diff)

# tests that if an edit is made, the new general format file has the edit
class TestEdit(unittest.TestCase):
    def test_edit(self):
        in_str = str(input_path + "/bounding/" + frame + "/boxes.json")
        out_str = str(output_path + "/bounding/" + frame + "/boxes.json")

        with open(in_str) as in_file:
            json_dict1 = json.load(in_file)
        with open(out_str) as out_file:
            json_dict2 = json.load(out_file)

        diff = DeepDiff(json_dict1, json_dict2, ignore_order=True)
        self.assertTrue(list(diff.keys()) == ['values_changed'] and len(diff) == 1, diff)

if __name__ == '__main__':
    unittest.main()