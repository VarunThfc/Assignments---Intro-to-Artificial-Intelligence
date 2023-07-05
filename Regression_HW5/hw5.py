import sys
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

if __name__ == "__main__":
    #2
    df = pd.read_csv(sys.argv[1])
    plt.figure()
    plt.plot(df['year'], df['days'])
    plt.xlabel('Year')
    plt.ylabel('Ice Days')
    plt.title('Year vs. Ice Days')

    plt.savefig('plot.jpg')

    #4
    df = df.to_numpy()
    x = df[:,0].reshape(-1,1)
    y = df[:,1].reshape(-1,1)

    #4.1
    x_ones = np.ones(x.shape)
    X = np.append(x_ones, x, axis=1)
    X = X.astype(np.int64)
    print('Q3a:')
    print(X)

    #4.2
    print('Q3b:')
    Y = y.T[0];
    print(Y)

    #4.3
    print('Q3c:')
    Z = X.T @ X
    print(Z)

    #4.4
    print('Q3d:')
    Z = np.linalg.inv(Z)
    print(Z)

    #4.5
    print('Q3e:')
    PI = Z @ X.T
    print(PI)

    #4.6
    print('Q3f:')
    hat_beta = PI @ y
    print(hat_beta.transpose()[0])

    #6
    #Q4

    x_test = 2022
    y_test = hat_beta[0] + hat_beta[1] * x_test
    print('Q4: ' + str(y_test[0]))


    # Q5
    if(hat_beta[1] > 0):
        print('Q5a: >')
        print('Q5b: Beta1 sign is positive, This parameter negitively weighs the year. The interpretation is as every year passes the number of snow days goes up by', hat_beta[1] ,'(because of the product')
    elif(hat_beta[1] == 0):
        print('Q5a: =')
        print('Q5b: Beta1 sign is 0, This outcome is independant of the year. The interpretation is as every year passes there is no impact on the on the ice days')
    else:
        print('Q5a: <')
        print('Q5b: Beta1 sign is negative, This parameter negitively weighs the year. The interpretation is as every year passes the number of snow days goes up by', hat_beta[1] ,'(because of the product')


    x0 = -1 * hat_beta[0] / hat_beta[1]
    print('Q6a:', x0[0])
    print('Q6b:  The model predicts that the lake will not freeze anymore by 2455 (midway), Due to lack of features within the data and using only year as a feature means naturally the model cannot generalize. Environmental prediction require lot of other features, examples temperature ranges during winter, gloabal trends, local weather pattern chages etc. Model assumes a linear realtion between ice days and year given the data which most likely cannot be extrapolated')