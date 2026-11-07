from data.data import prepare_data


def main():
    
    train_df, _ = prepare_data()
    
    label_counts = train_df["label"].value_counts()
    
    print(f"Number of intent class:{label_counts.size}")
    print(f"Total  training examples: {len(train_df)}")
    print("\nExamples per intent:")
    print(label_counts.to_string())
    
    print("--" * 50)
    
    print(f"Most represented intent: {label_counts.idxmax()} -> {label_counts.max()}")
    print(f"Least represented intent: {label_counts.idxmin()} -> {label_counts.min()}")
    print(f"Imbalancee ratio: {label_counts.max() / label_counts.min():.2f}")
    
    print("--" * 50)
    
    print(f"Mean examples per intent: {label_counts.mean():.2f}")
    print(f"Median examples per intent: {label_counts.median():.2f}")
    



if __name__ == "__main__":
    main()