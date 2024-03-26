import pandas as pd
import glob

def process_and_combine_csv(type):
    # Build the file pattern based on the file_type argument
    pattern = f'results/benchmark_{type}_*.csv'
    csv_files = glob.glob(pattern)
    df_list = [pd.read_csv(file) for file in csv_files if csv_files]

    if df_list:
        combined_df = pd.concat(df_list, ignore_index=True)
        grouped_df = combined_df.groupby(['n_samples', 'n_epochs'], as_index=False).mean()
        output_file = f'results/benchmarks_{type}_averaged.csv'
        grouped_df.to_csv(output_file, index=False)
        print(f"Averaged data saved to {output_file}")
        return output_file
    else:
        print(f"No CSV files matched the pattern {pattern}.")
        return None

# Process and save averaged benchmarks for both mojo and py by calling the function with the respective type
mojo_file = process_and_combine_csv('mojo')
py_file = process_and_combine_csv('py')

# Function to combine the averaged benchmark files of mojo and py into one
def combine_averaged_benchmarks(mojo_file, py_file):
    if mojo_file and py_file:
        df_mojo = pd.read_csv(mojo_file)
        df_py = pd.read_csv(py_file)
        
        # Rename columns to indicate type
        df_mojo.rename(columns={'time': 'time_mojo', 'accuracy': 'accuracy_mojo'}, inplace=True)
        df_py.rename(columns={'time': 'time_py', 'accuracy': 'accuracy_py'}, inplace=True)
        
        # Merge the two DataFrames on 'n_samples' and 'n_epochs'
        combined_df = pd.merge(df_mojo, df_py, on=['n_samples', 'n_epochs'])
        
        # Reorder columns to have time values next to each other, then accuracy values
        combined_df = combined_df[['n_samples', 'n_epochs', 'time_py', 'time_mojo', 'accuracy_py', 'accuracy_mojo']]
        
        # Save the combined DataFrame to a new CSV file
        combined_df.to_csv('results/benchmarks_comparison.csv', index=False)
        print("Combined averaged benchmarks saved to results/benchmarks_comparison.csv")
    else:
        print("Could not find both mojo and py benchmark files.")

# Combine the mojo and py benchmarks into one file
combine_averaged_benchmarks(mojo_file, py_file)
