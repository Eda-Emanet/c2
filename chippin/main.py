# Example usage
def main():
    # Initialize processor with data paths
    processor = DataProcessor(
        best_path='C:/Users/eda.emanet/Chippin/cust_best_sample.csv',
        cust_path='C:/Users/eda.emanet/Chippin/cust_sample.csv',
        trx_path='C:/Users/eda.emanet/Chippin/trx_sample.csv'
    )
    
    # Process the data
    processor.process_data()
    
    # Get the RFM results
    rfm_results = processor.get_rfm_data()
    
    # Display the first few rows of results
    print(rfm_results.head())
    
    # Optional: create visualizations
    processor.plot_gender_distribution()
    processor.plot_birth_date_status()

if __name__ == "__main__":
    main()