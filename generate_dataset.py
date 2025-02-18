import os

def process_file(
    input_file: str, 
    output_file: str
) -> None:
    """
    Process the input TSV file and write the first column to the output file.

    Args:
        input_file (str): Path to the input TSV file.
        output_file (str): Path to the output text file.

    Returns:
        None
    """
    try:
        with open(input_file, "r", encoding="utf-8") as in_file, \
             open(output_file, "w", encoding="utf-8") as out_file:
            for line in in_file:
                # Split the line by tab and write only the first column
                processed_line = line.split("\t")[0] + " "
                out_file.write(processed_line)
    except IOError as e:
        print(f"An error occurred while processing the file: {e}")

# Get the current working directory
curr_path = os.getcwd()

# Define input and output file paths
input_file = os.path.join(curr_path, "dataset_files", "pmindia.v1.hi-en.tsv")
output_file = os.path.join(curr_path, "dataset_files", "dataset.txt")

# Process the file
process_file(input_file, output_file)
