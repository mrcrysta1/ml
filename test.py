import pandas as pd

def process_csv(filename):
  """
  Reads a CSV file, removes the last 3 columns, and iterates over the data.

  Args:
      filename (str): The name of the CSV file.
  """

  while True:
    try:
      # Read the CSV file, dropping last 3 columns
      data = pd.read_csv(filename, usecols=lambda x: x[:-3])

      # Check if there's data in the DataFrame. If not, break the loop.
      if data.empty:
        break

      # Process the data (you can replace this with your specific logic)
      for index, row in data.iterrows():
        print(f"Row {index+1}: {row.to_list()}")

    except FileNotFoundError:
      print(f"Error: File '{filename}' not found.")
      break

# Example usage
filename = "train_datasets.csv"  # Assuming the extra ".csv" was a mistake

# Example usage
process_csv(filename)
