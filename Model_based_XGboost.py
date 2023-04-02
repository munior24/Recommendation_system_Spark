import sys
import json
from time import time
import numpy as np
import xgboost as xgb
from pyspark import SparkContext

def load_data(sc, folder_path, train_file, test_file):
    pre_train_data = sc.textFile(train_file)\
        .filter(lambda x: x != "user_id,business_id,stars").map(lambda x: x.split(','))

    user_to_train = set(pre_train_data.map(lambda x: x[0]).distinct().collect())
    business_to_train = set(pre_train_data.map(lambda x: x[1]).distinct().collect())

    user_feature_map = sc.textFile(folder_path + 'user.json').map(lambda x: json.loads(x)) \
        .filter(lambda x: x['user_id'] in user_to_train) \
        .map(lambda x: (x['user_id'], [x['review_count'], x['average_stars']])) \
        .collectAsMap()

    business_feature_map = sc.textFile(folder_path + 'business.json').map(lambda x: json.loads(x)) \
        .filter(lambda x: x['business_id'] in business_to_train) \
        .map(lambda x: (x['business_id'], [x['review_count'], x['stars']])) \
        .collectAsMap()

    x_train = np.array(pre_train_data.map(lambda x: np.array([user_feature_map[x[0]], business_feature_map[x[1]]]).flatten()).collect())
    y_train = np.array(pre_train_data.map(lambda x: float(x[2])).collect())
    x_test = np.array(sc.textFile(test_file).filter(lambda x: x != "user_id,business_id,stars").map(lambda x: x.split(','))\
        .map(lambda x: np.array([user_feature_map.get(x[0], [0, 2.5]),business_feature_map.get(x[1], [0, 2.5])]).flatten()).collect())
    test_rdd = sc.textFile(test_file).filter(lambda x: x != "user_id,business_id,stars").map(lambda x: x.split(','))

    return x_train, y_train, x_test, test_rdd

def write_output(output_file_name,res_rdd, predicted_value ):
    with open(output_file_name, 'w') as f:
        f.write('user_id,business_id,prediction\n')
        for pair in zip(res_rdd.collect(), predicted_value):
            f.write(pair[0][0] + "," + pair[0][1] + "," + str(pair[1]) + "\n")
        

    
def main():
    start = time()
    sc = SparkContext.getOrCreate()

    folder_path = sys.argv[1]+'/'
    train_file = folder_path + 'yelp_train.csv'
    test_file = sys.argv[2]

    x_train, y_train, x_test, test_rdd = load_data(sc, folder_path, train_file, test_file)

    # Do something with the data

    output_file_name = sys.argv[3]
    
    modl = xgb.XGBRegressor(objective = 'reg:linear', n_estimators=100, max_depth=5, n_jobs=-1)
    modl.fit(x_train,y_train)
    predicted_value = modl.predict(x_test)
    res_rdd = test_rdd.map(lambda x: (x[0], x[1]))
    write_output(output_file_name,res_rdd, predicted_value )
    end = time()
    print("Duration :  ", (end-start))
    


if __name__ == "__main__":
    main()
