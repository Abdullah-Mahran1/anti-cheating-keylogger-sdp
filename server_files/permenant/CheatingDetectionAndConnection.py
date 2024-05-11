import time

# Define the paths of the student's files, microcontroller id file, accuracy file, and the instructor's interface file
student_file_paths = [
    r"D:\arduino senior project\202001805\StudentData.txt",
    r"D:\arduino senior project\202001806\StudentData.txt",
    r"D:\arduino senior project\201804181\StudentData.txt",   
    r"D:\arduino senior project\201906710\StudentData.txt",   
    r"D:\arduino senior project\201912079\StudentData.txt",
    r"D:\arduino senior project\202001804\StudentData.txt",
]
microcontroller_id_file = r"D:\arduino senior project\microcontrollerid.txt"
instructor_file_path = r"D:\arduino senior project\instructorinterface.txt"
accuracy_file_path = r"D:\arduino senior project\permenant\Cheating_Analysis.txt"

# Define the keywords to search for
keywords = ['chegg', 'chatgpt', 'quizlet', 'outlook', 'Chegg', 'Chatgpt', 'Quizlet', 'Outlook', 'Chat', 'chat', 'Whats', 'whats']

# Function to read microcontroller IDs and their connection status
def read_microcontroller_ids(file_path):
    student_statuses = {}
    try:
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split(',')
                if len(parts) == 2:
                    student_statuses[parts[1].strip()] = parts[0].strip()  # ID as key, status as value
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    return student_statuses

# Function to read accuracy data
def read_accuracy_data(file_path):
    accuracy_data = {}
    try:
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split(',')
                if len(parts) == 2:
                    exam_id, accuracy = parts
                    accuracy_data[exam_id.strip()] = accuracy.strip()
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    return accuracy_data

# Function to check for cheating attempt
def check_for_cheating(file_path, keywords):
    try:
        with open(file_path, 'r') as file:
            content = file.read().strip()
            if not content:  # If the content is empty
                return 'disconnected', False, None
            for keyword in keywords:
                if keyword in content.lower():
                    return 'connected', True, keyword
            return 'connected', False, None
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return 'disconnected', False, None

# Function to write the result to the instructor's interface file
def write_results_to_interface(results, accuracy_data):
    try:
        with open(instructor_file_path, 'w') as file:
            for exam_id, status, cheating_detected, detected_keyword in results:
                accuracy = accuracy_data.get(exam_id, 'accuracy:0%')
                cheating_status = f", cheating attempt detected: '{detected_keyword}'" if cheating_detected else ""
                line = f"{exam_id}, {status}, {accuracy}{cheating_status}\n"  # Add accuracy before cheating status
                file.write(line)
    except FileNotFoundError:
        print("File not found.")

# Continuously monitor the files for cheating attempts and students' connection status
while True:
    student_statuses = read_microcontroller_ids(microcontroller_id_file)
    accuracy_data = read_accuracy_data(accuracy_file_path)
    results = []
    for student_file_path in student_file_paths:
        exam_id = student_file_path.split('\\')[-2]  # Extract ID from the file path
        connection_status = 'disconnected'
        if exam_id in student_statuses and student_statuses[exam_id] != 'none':
            connection_status = 'connected'  # If ID found and status is not 'none'
        cheating_status, cheating_detected, detected_keyword = check_for_cheating(student_file_path, keywords)
        results.append((exam_id, connection_status, cheating_detected, detected_keyword))
        if connection_status == 'disconnected':
            print(f"{exam_id} {connection_status}")
        elif cheating_detected:
            print(f"{exam_id} {connection_status} cheating attempt detected: '{detected_keyword}'")
        else:
            print(f"{exam_id} {connection_status}")
    write_results_to_interface(results, accuracy_data)
    time.sleep(5)  # Check every 5 seconds