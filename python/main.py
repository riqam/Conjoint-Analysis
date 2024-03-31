import pandas as pd
import numpy as np
import joblib

def preprocess_data():
    # baca file experimental design
    df = pd.read_csv('data\experimental_design.csv')

    # Mengambil kolom 'skill', 'bentuk_program', dan 'harga_program' 30 baris pertama
    df_set = df.loc[0:29, ['skill', 'bentuk_program', 'harga_program']]

    df = pd.get_dummies(df, columns=['skill', 'bentuk_program'])
    df = df.replace({False: 0, True: 1})
    df = df.drop(columns=['participant', 'set', 'alternatif'])

    return df, df_set

# fungsi untuk normalisasi data
def min_max_normalize(X):
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_normalized = (X - X_min) / (X_max - X_min)

    return X_normalized

# split X dan y
def split(df):
    X = df.drop(columns=['pilihan'])
    y = df['pilihan']

    return X, y

# buat model logistic regression
def logistic_regression(X, y, learning_rate=0.01, iterations=10000):
    # inisiasi nilai koefisien awal dengan 0
    beta = np.zeros(X.shape[1])
    
    for _ in range(iterations):
        z = np.dot(X, beta)
        pi_x = np.exp(z) / (1 + np.exp(z))
        
        # perbaharui koefisien
        gradient = np.dot(X.T, pi_x - y) / y.size
        beta -= learning_rate * gradient
    
    return beta

# save nilai beta (utility score untuk setiap level)
def save_beta():
    beta_df = pd.DataFrame({
    'Atribute Level': X.columns,
    'Utility Score': beta_estimates
    })

    joblib.dump(beta_df, 'beta_df.pkl')

    return beta_df

# hitung total utility score
def total_util():
    df_set_enc = pd.get_dummies(df_set, columns=['skill', 'bentuk_program'])
    df_set_enc['harga_program'] = min_max_normalize(df_set['harga_program'])

    df_set_aligned = pd.DataFrame(columns=X.columns, data=np.zeros((df_set_enc.shape[0], X.columns.size)))

    for col in df_set_enc.columns:
        if col in df_set_aligned.columns:
            df_set_aligned[col] = df_set_enc[col]

    # hitunt total utility untuk setiap pilihan
    df_set_aligned['total_utility_score'] = np.dot(df_set_aligned.values, beta_estimates)
    df_set_aligned['choice_probability'] = df_set_aligned['total_utility_score'].apply(lambda x: (np.exp(x) / (1 + np.exp(x)))*100)
    df_set_aligned = df_set_aligned[['total_utility_score', 'choice_probability']]

    return df_set_aligned

# simpan hasil final
def final_result():
    result = pd.concat([df_set, df_set_aligned], axis=1)
    best_result = result.nlargest(5, 'choice_probability')

    joblib.dump(result, 'result.pkl')
    joblib.dump(best_result, 'best_result.pkl')

    return result, best_result

if __name__ == '__main__':
    # preprocess data
    df, df_set = preprocess_data()

    # normalisasi data
    df['harga_program'] = min_max_normalize(df['harga_program'])

    # inisiasi X dan y
    X, y = split(df)

    # inisiasi nilai intercept 
    X['Intercept'] = 1

    # covert dataframe ke numpy array untuk fitting model
    X_np = X.values
    y_np = y.values

    # terapkan model lr
    beta_estimates = logistic_regression(X_np, y_np)

    # simpan utility score
    beta_df = save_beta()

    # hitung total utility
    df_set_aligned = total_util()

    # simpan final result
    result, best_result = final_result()
