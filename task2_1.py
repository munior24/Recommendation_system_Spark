from collections import defaultdict
from pyspark import SparkConf, SparkContext
import sys
import time
import math
from itertools import combinations


weights_dict = {}




def getWeight(test_business_id, train_business_id):
    if str(test_business_id)+"_"+str(train_business_id) in weights_dict.keys():
        weight = weights_dict[str(test_business_id)+"_"+str(train_business_id)]
    elif str(train_business_id)+"_"+str(test_business_id) in weights_dict.keys():
        weight = weights_dict[str(train_business_id)+"_"+str(test_business_id)]
    else:
        weight = getSimilarity((test_business_id,train_business_id), ratings)
        weight = weight*pow(abs(weight),2)
    weights_dict[str(train_business_id)+"_"+str(test_business_id)] = weight
    return weight

def sorting(x):
    x = list(x)
    return sorted(x,key=lambda y: y[1],reverse=True)[0:min(len(x),100)]

def write_out(output_file, predicted_ratings):
    with open(output_file, "w") as f:
            f.write("user_id,business_id,prediction\n")
            for pred in predicted_ratings:
                f.write(pred[0][0]+","+pred[0][1]+","+str(pred[1]))
                f.write("\n")

def toList(a):
    return [a]

def appnd(a, b):
    a.append(b)
    return a

def ext(a, b):
    a.extend(b)
    return a

def getSimilarity(pair,ratings):
    try:
        item1_users = set(ratings[pair[0]].keys())
        items2_users = set(ratings[pair[1]].keys())

        co_rated_users = set(item1_users) & set(items2_users)
        if(len(co_rated_users)>9):
            item1_ratings = []
            item2_ratings = []  

            for user in co_rated_users:
                item1_ratings.appnd(float(ratings[pair[0]][user]))
                item2_ratings.appnd(float(ratings[pair[1]][user]))
            
            item1_average = sum(item1_ratings)/len(item1_ratings)
            item2_average = sum(item2_ratings)/len(item2_ratings)

            item1_rating_average = [i-item1_average for i in item1_ratings]
            item2_rating_average = [i-item2_average for i in item2_ratings]

            numerator = sum([item1_rating_average[i]*item2_rating_average[i] for i in range(len(item1_rating_average))])
            denominator = math.sqrt(sum([i*i for i in item1_rating_average])) * math.sqrt(sum([i*i for i in item2_rating_average]))

            if numerator==0 or denominator==0:
                return 0.5
            return numerator/denominator

        else: 
            return 0.5
    except:
        return 0.5          
            
if __name__ == '__main__':
    # File paths
    train_file_path = sys.argv[1]
    test_file_path = sys.argv[2]
    output_file_path = sys.argv[3]

    # SparkContext initialization
    sc = SparkContext()

    # Start time
    start_time = time.time()

    # Load train file and filter header row
    train_file = sc.textFile(train_file_path)
    header_row = train_file.first()

    # Extract reviews from train file
    train_reviews = train_file.filter(lambda row: row != header_row).map(lambda row: row.split(",")).map(lambda review: (review[1], review[0], review[2])).distinct().persist()

    # Group businesses and user IDs from train reviews
    business_user_ids_list = train_reviews.map(lambda review: (review[0], review[1])).combineByKey(toList, appnd, ext).persist()
    filtered_businesses = business_user_ids_list.map(lambda x: x[0]).distinct().collect()

    # Extract ratings from train reviews and store them in a dictionary
    ratings = train_reviews.map(lambda review: (review[0], (review[1], review[2]))).combineByKey(toList, appnd, ext).map(lambda x: (x[0], dict(list(set(x[1]))))).collectAsMap()

    # Load test file and filter header row
    test_file = sc.textFile(test_file_path)
    header_test = test_file.first()

    # Group businesses and user IDs from test file
    test_business_user_ids_list = test_file.filter(lambda row: row != header_test).map(lambda row: row.split(",")).map(lambda x: (x[0], x[1])).zipWithIndex().persist()

    # Group user IDs by business ID from train reviews and store them in a dictionary
    user_business_ids_dict = train_reviews.map(lambda review: (review[1], review[0])).combineByKey(toList, appnd, ext).map(lambda x: (x[0], set(x[1]))).persist().collectAsMap()

    # Generate predictions for test reviews
    predictions = test_business_user_ids_list.map(lambda x: ((x[0][0], x[0][1], x[1]), user_business_ids_dict[x[0][0]])).map(lambda x: [(x[0], y) for y in x[1]]).flatMap(lambda x: x).map(lambda x: (x[0], (ratings[x[1]][x[0][0]], getWeight(x[0][1], x[1])))).groupByKey().mapValues(sorting).map(lambda x: (x[0], sum([w*float(r) for r, w in x[1]])/sum([abs(w) for _, w in x[1]]))).collect()

    # Sort predictions by review ID
    sorted_predictions = sorted(predictions, key=lambda x: (x[0][2]))

    # Write predictions to output file
    write_out(output_file_path, sorted_predictions)

    # Print elapsed time
    print(time.time() - start_time)

    # Stop SparkContext
    sc.stop()
