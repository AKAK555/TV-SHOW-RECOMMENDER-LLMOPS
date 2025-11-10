import pandas as pd

class TVShowDataLoader:
    def __init__(self, original_csv:str, processed_csv:str):
        self.original_csv = original_csv
        self.processed_csv = processed_csv

    def load_and_process(self):
        """Load TV show data from a CSV file."""
        try:
            df = pd.read_csv(self.original_csv, encoding='utf-8', on_bad_lines='skip').dropna()
            required_cols = ['name','genres','overview'] 
            missing = required_cols - set(df.columns)
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
            
            df['combined_info'] = (
                "Title: " + df['name'] + 
                ".. Genres: " + df['genres'] + 
                ".. Overview: " + df['overview']
            )
            
            df[['combined_info']].to_csv(self.processed_csv, index=False, encoding='utf-8')
            
            return self.processed_csv
        
        except FileNotFoundError:
            print(f"Error: The file at {self.file_path} was not found.")
            return pd.DataFrame()
        except pd.errors.EmptyDataError:
            print("Error: The file is empty.")
            return pd.DataFrame()
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return pd.DataFrame()