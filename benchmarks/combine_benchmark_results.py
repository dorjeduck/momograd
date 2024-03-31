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
momograd_file = process_and_combine_csv('momograd')
momogradx_file = process_and_combine_csv('momogradx')
micrograd_file = process_and_combine_csv('micrograd')

# Function to combine the averaged benchmark files of mojo and py into one
def combine_averaged_benchmarks(momograd_file, momogradx_file,micrograd_file):
    if momograd_file and micrograd_file and momogradx_file:
        df_momograd = pd.read_csv(momograd_file)
        df_momogradx = pd.read_csv(momogradx_file)
        df_micrograd = pd.read_csv(micrograd_file)
        
        # Rename columns to indicate type
        df_momograd.rename(columns={'time': 'time_momograd', 'accuracy': 'accuracy_momograd'}, inplace=True)
        df_momogradx.rename(columns={'time': 'time_momogradx', 'accuracy': 'accuracy_momogradx'}, inplace=True)
        df_micrograd.rename(columns={'time': 'time_micrograd', 'accuracy': 'accuracy_micrograd'}, inplace=True)
        
        # First merge df_momograd and df_momogradx
        combined_df = pd.merge(df_momograd, df_momogradx, on=['n_samples', 'n_epochs'])
        
        # Then merge the result with df_micrograd
        combined_df = pd.merge(combined_df, df_micrograd, on=['n_samples', 'n_epochs'])
        
        # Reorder columns to have time values next to each other, then accuracy values
        combined_df = combined_df[['n_samples', 'n_epochs', 'time_micrograd', 'time_momograd','time_momogradx', 'accuracy_micrograd', 'accuracy_momograd', 'accuracy_momogradx']]
        
        # Save the combined DataFrame to a new CSV file
        combined_df.to_csv('results/benchmarks_comparison.csv', index=False)
        print("Combined averaged benchmarks saved to results/benchmarks_comparison.csv")
    else:
        print("Could not find both mojo and py benchmark files.")

# Combine the mojo and py benchmarks into one file
combine_averaged_benchmarks(momograd_file,momogradx_file, micrograd_file)
