import pandas as pd

def train_test_split(df, frac=0.2):
    
    # get random sample 
    test = df.sample(frac=frac, axis=0,  random_state = 42)

    # get everything but the test sample
    train = df.drop(index=test.index)

    return train, test


df = pd.read_csv('/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/classifier/csv_data/alz_data.csv')
tra, test = train_test_split(df)
print(len(tra))
print(len(test))
print(test.Label.sum()/len(test))
print(1-test.Label.sum()/len(test))