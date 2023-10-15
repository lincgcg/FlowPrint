

    # Load flows from file 'flows.p'
    X, y = preprocessor.load('/Users/cglin/Desktop/flows.p')
    # Split data into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    
    # Create FlowPrint object
    flowprint = FlowPrint(
        batch       = 300,
        window      = 30,
        correlation = 0.1,
        similarity  = 0.9
    )

    # Fit FlowPrint with flows and labels
    flowprint.fit(X_train, y_train)

    # Create fingerprints for test data
    fp_test = flowprint.fingerprint(X_test)
    # Predict best matching fingerprints for each test fingerprint
    y_pred = flowprint.recognize(fp_test)

    # Print report with 4 digit precision
    print(classification_report(y_test, y_pred, digits=4))