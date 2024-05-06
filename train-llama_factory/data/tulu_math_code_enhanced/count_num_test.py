import json

# count how many data in the json file
def count_num(input_file):
    with open(input_file, 'r') as file:
        data = json.load(file)
        print(len(data))

if __name__ == '__main__':
    count_num('tulu2_math_code_enhanced.json')

