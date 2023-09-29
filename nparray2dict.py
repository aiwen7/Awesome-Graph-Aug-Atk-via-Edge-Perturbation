import numpy as np
import pickle
import collections
'''
def convert_and_save(adj_full_numpy, dataset_name):
    output_list = []
    afn = adj_full_numpy
    row_col_tuple = np.where(afn > 0)
    row = row_col_tuple[0]
    col = row_col_tuple[1]
    for i in range(len(row)):
        output_list.append((row[i], col[i]))
    output_dict = collections.defaultdict(list)
    for key, value in output_list:
        output_dict[key].append(value)
    #print(output_dict)
    print(output_dict)
    write_file = open("ind." + dataset_name + ".graph", "wb")
    pickle.dump(output_dict, write_file)
    write_file.close()
'''
def convert_and_save(adj_ori, adj_full_numpy, dataset_name):
    output_list = []
    afn = adj_full_numpy
    row_col_tuple = np.where(afn > 0)
    row = row_col_tuple[0]
    col = row_col_tuple[1]
    for i in range(len(row)):
        output_list.append((row[i], col[i]))
    output_dict = collections.defaultdict(list)
    for key, value in output_list:
        output_dict[key].append(value)
    #print(output_dict)
    #for i in output_dict:
    #    print(i)
    write_file = open("ind." + dataset_name + ".graph", "wb")
    pickle.dump(output_dict, write_file)
    write_file.close()
    output_list_ori = []
    afn_ori = adj_ori
    row_col_tuple_ori = np.where(afn_ori > 0)
    row_ori = row_col_tuple_ori[0]
    col_ori = row_col_tuple_ori[1]
    for i in range(len(row_ori)):
        output_list_ori.append((row_ori[i], col_ori[i]))
    output_dict_ori = collections.defaultdict(list)
    for key, value in output_list_ori:
        output_dict_ori[key].append(value)
    diff = output_dict_ori.keys() & output_dict
    diff_vals = [(k, output_dict_ori[k], output_dict[k]) for k in diff if output_dict_ori[k] != output_dict[k]]
    print(diff_vals)
    missing_keys = []
    missing_values = {}
    for key in output_dict_ori:
        if key not in output_dict:
            missing_keys.append(key)
            missing_values[key] = output_dict_ori[key]
    print("Missing keys are:",missing_keys)
    print("Missing values are:",missing_values)
def main():
    temp_array = np.array([[0.0, 5.0, 0.0, 3.0, 1.0, 2.0, 0.0, 1.0, 2.0, 1.0, 0.0], \
                           [9.0, 0.0, 0.0, 0.0, 2.0, 1.0, 0.0, 3.0, 0.0, 4.0, 5.0], \
                           [0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 3.0, 1.0, 2.0, 0.0, 0.0], \
                           [1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 3.0, 3.0, 0.0, 2.0, 0.0], \
                           [0.0, 0.0, 3.0, 2.0, 2.0, 1.0, 0.0, 0.0, 1.0, 0.0, 6.0]])
    convert_and_save(temp_array, "testadj")

if __name__ == "__main__":
    main()