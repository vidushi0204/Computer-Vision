import sys
from task1 import task1
from task2 import task2

def main():
    if len(sys.argv) < 4:
        print("Usage: python3 main.py <task_number> <input_img_dir> <output_img_dir_or_csv>")
        sys.exit(1)
    
    task_number = sys.argv[1]
    input_dir = sys.argv[2]
    output = sys.argv[3]
    
    if task_number == '1':
        task1(input_dir, output)
    elif task_number == '2':
        task2(input_dir, output)
if __name__ == "__main__":
    main()
