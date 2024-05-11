# import os
# import time
# import subprocess

# def read_student_keystrokes(filepath):
#     keystrokes = []
#     with open(filepath, 'r') as file:
#         for line in file:
#             keystrokes.append(line.strip())
#     return keystrokes

# def clear_student_keystrokes(filepath, count):
#     with open(filepath, 'r') as file:
#         lines = file.readlines()
#     with open(filepath, 'w') as file:
#         file.writelines(lines[count:])

# def read_do_train(filepath):
#     train_dict = {}
#     with open(filepath, 'r') as file:
#         for line in file:
#             student_id, status = line.strip().split(',')
#             train_dict[student_id] = status
#     return train_dict

# def write_do_train(filepath, train_dict):
#     with open(filepath, 'w') as file:
#         for student_id, status in train_dict.items():
#             file.write(f"{student_id},{status}\n")
# def execute_2convert(doTrain):
#     print(f"Executing model with attribute: {doTrain}, cwd: {os.path.dirname(__file__)}")
#     subprocess.run(["python", os.path.join(os.path.dirname(__file__),"assemble.py")])
#     subprocess.run(["python",os.path.join(os.path.dirname(__file__),"3model.py"), str(doTrain), "202001804"])
#     print(f"Executed model with attribute: {doTrain}")

# def main():
#     keystrokes_filepath = r"D:\arduino senior project\202001804\Studentkeystrocks.txt"
#   #  keystrokes_username = keystrokes_filepath.split("\\")
#   #  keystrokes_username = keystrokes_username[len(keystrokes_username-2)]
#     do_train_filepath = r"D:\arduino senior project\do_train.txt"
    
#     while True:
#         keystrokes = read_student_keystrokes(keystrokes_filepath)
#        # print(f'42: {len(keystrokes)}')
#         if len(keystrokes) >= 100:
#             print("hi")
#             train_dict = read_do_train(do_train_filepath)
#             student_id = os.path.basename(os.path.dirname(keystrokes_filepath))

#             if train_dict.get(student_id) == 'none':
#                 execute_2convert("0")
#              #   clear_student_keystrokes(keystrokes_filepath, 100)
#             elif train_dict.get(student_id) == 'train' and len(keystrokes) >= 1000:
#                 execute_2convert("1")
#              #   clear_student_keystrokes(keystrokes_filepath, 500)
#                 train_dict[student_id] = 'none'
#                 write_do_train(do_train_filepath, train_dict)
        
#         time.sleep(1)  # Adjust sleep time as needed

# # Call the main function directly
# main()

import os
import time
import subprocess

def read_student_keystrokes(filepath):
    keystrokes = []
    with open(filepath, 'r') as file:
        for line in file:
            keystrokes.append(line.strip())
    return keystrokes

def clear_student_keystrokes(filepath, count):
    with open(filepath, 'r') as file:
        lines = file.readlines()
    with open(filepath, 'w') as file:
        file.writelines(lines[count:])

def read_do_train(filepath):
    train_dict = {}
    with open(filepath, 'r') as file:
        for line in file:
            student_id, status = line.strip().split(',')
            train_dict[student_id] = status
    return train_dict

def write_do_train(filepath, train_dict):
    with open(filepath, 'w') as file:
        for student_id, status in train_dict.items():
            file.write(f"{student_id},{status}\n")

def execute_2convert(doTrain, student_id):
    # Replace this with your code to execute 3model.py
    print(f"Executing 3model.py with attribute: {doTrain} by {student_id}")
    subprocess.run(["python", os.path.join(os.path.dirname(__file__),"assemble.py")])
    subprocess.run(["python", os.path.join(os.path.dirname(__file__),"3model.py"), str(doTrain), student_id])


def main():
    students_base_path = r"D:\arduino senior project"
    do_train_filepath = os.path.join(students_base_path, "do_train.txt")

    while True:
        with open(do_train_filepath, 'r') as do_train_file:
            for line in do_train_file:
                student_id, status = line.strip().split(',')
                student_keystrokes_filepath = os.path.join(students_base_path, student_id, "Studentkeystrocks.txt")
                keystrokes = read_student_keystrokes(student_keystrokes_filepath)

                if status == 'none' and len(keystrokes) >= 100:
                    execute_2convert("0", student_id)
                   # clear_student_keystrokes(student_keystrokes_filepath, 100)
                elif status == 'train' and len(keystrokes) >= 1000:
                    execute_2convert("1", student_id)
                   # clear_student_keystrokes(student_keystrokes_filepath, 500)
                    # Update status to 'none' in do_train.txt
                    train_dict = read_do_train(do_train_filepath)
                    train_dict[student_id] = 'none'
                    write_do_train(do_train_filepath, train_dict)
        
        time.sleep(1)  # Adjust sleep time as needed


main()


