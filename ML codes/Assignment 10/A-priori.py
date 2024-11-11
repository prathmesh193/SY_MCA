from itertools import combinations

# Input transactions
transactions = [
    ['I1', 'I2', 'I3'],
    ['I2', 'I3', 'I4'],
    ['I4', 'I5'],
    ['I1', 'I2', 'I4'],
    ['I1', 'I2', 'I3', 'I5'],
    ['I1', 'I2', 'I3', 'I4']
]

# Set the support and confidence thresholds
support_threshold = 0.5
confidence_threshold = 0.6

# Function to calculate support for an itemset
def calculate_support(itemset, transactions):
    count = sum(1 for transaction in transactions if set(itemset).issubset(transaction))
    return count / len(transactions)

# Find frequent itemsets based on support
def find_frequent_itemsets(transactions, support_threshold):
    items = {item for transaction in transactions for item in transaction}
    frequent_itemsets = []
    
    # Generate itemsets of increasing size
    size = 1
    while True:
        itemsets = [list(comb) for comb in combinations(items, size)]
        # Filter itemsets by support
        itemsets = [itemset for itemset in itemsets if calculate_support(itemset, transactions) >= support_threshold]
        
        if not itemsets:  # Stop if no frequent itemsets are found
            break
        
        frequent_itemsets.extend(itemsets)
        size += 1

    return frequent_itemsets

# Generate association rules from frequent itemsets
def generate_rules(frequent_itemsets, transactions, confidence_threshold):
    rules = []
    for itemset in frequent_itemsets:
        if len(itemset) < 2:
            continue
        # Generate rules for subsets of the itemset
        for i in range(1, len(itemset)):
            for antecedent in combinations(itemset, i):
                antecedent = list(antecedent)
                consequent = [item for item in itemset if item not in antecedent]
                
                support_itemset = calculate_support(itemset, transactions)
                support_antecedent = calculate_support(antecedent, transactions)
                confidence = support_itemset / support_antecedent
                
                if confidence >= confidence_threshold:
                    rules.append((antecedent, consequent, confidence))
    return rules

# Run the simplified Apriori algorithm
frequent_itemsets = find_frequent_itemsets(transactions, support_threshold)
rules = generate_rules(frequent_itemsets, transactions, confidence_threshold)

# Output results
print("Frequent Itemsets:")
for itemset in frequent_itemsets:
    print(itemset)

print("\nAssociation Rules:")
for antecedent, consequent, confidence in rules:
    print(f"{antecedent} => {consequent} (Confidence: {confidence:.2f})")
