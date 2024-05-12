import os
import random

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.ml.feature import MinHashLSH, CountVectorizer
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import BucketedRandomProjectionLSH
from pyspark.sql.functions import collect_list, col, size, corr, rand


def main(spark, userID):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    userID : string, userID of student to find files in HDFS
    '''

    # Load the boats.txt and sailors.json data into DataFrame
    ratings = spark.read.csv(f'hdfs:/user/{userID}/ml-latest-small/ratings.csv', header = True, schema='userId string, movieId string, rating FLOAT, timestamp INT')
    # movies = spark.read.csv(f'hdfs:/user/{userID}/ml-latest-small/movies.csv')
    # links = spark.read.csv(f'hdfs:/user/{userID}/ml-latest-small/links.csv')
    # tags = spark.read.csv(f'hdfs:/user/{userID}/ml-latest-small/tags.csv')

    # tr, val, tst = train_val_test_split(ratings)
    # tr.orderBy('userId').show()
    # val.orderBy('userId').show()
    # tst.orderBy('userId').show()

    print('Printing ratings with specified schema')
    ratings.printSchema()
    user_movies = ratings.groupBy("userId").agg(collect_list("movieId").alias("movies"))
    user_movies = user_movies.filter(size("movies") >= 5)

    # Convert movieId lists to sparse vector
    vectorizer = CountVectorizer(inputCol="movies", outputCol="features")
    model = vectorizer.fit(user_movies)
    user_movies_vector = model.transform(user_movies)

    mh = MinHashLSH(inputCol="features", outputCol="hashes", numHashTables=5)
    model = mh.fit(user_movies_vector)
    movie_twins = model.approxSimilarityJoin(user_movies_vector, user_movies_vector, threshold=1.0, distCol="JaccardDistance")
    movie_twins = movie_twins.filter("datasetA.userId < datasetB.userId")

    simplified_df = movie_twins.select(
            col("datasetA.userId").alias("userIdA"),
            col("datasetB.userId").alias("userIdB"),
            "JaccardDistance"
        )

    simplified_df = simplified_df.orderBy(col("JaccardDistance"), col("userIdA"), col("userIdB"))
    top_100_twins = simplified_df.limit(100)
    top_100_twins.show()

    # top_100_twins.write.csv("./top100_twins.csv")

    print("___Q2___")
    ratings_A = top_100_twins.alias('pairs').join(ratings, col('pairs.userIdA') == col('userId')).select(col('pairs.userIdA'), 
        col('pairs.userIdB'), 
        col('movieId'), 
        col('rating').alias('rating_A')
    )
    ratings_B = top_100_twins.alias('pairs').join(ratings, col('pairs.userIdB') == col('userId')).select(
        col('pairs.userIdA'), 
        col('pairs.userIdB'), 
        col('movieId'), 
        col('rating').alias('rating_B')
    )

    # Join on movieId to compare only ratings for the same movies
    paired_ratings = ratings_A.join(ratings_B, on=['userIdA', 'userIdB', 'movieId'])
    correlation_data = paired_ratings.groupBy('userIdA', 'userIdB').agg(corr('rating_A', 'rating_B').alias('correlation'))
    average_correlation = correlation_data.agg({'correlation': 'avg'}).collect()[0][0]
    # average_correlation.show()

    # Random pairs
    user_ids = ratings.select('userId').distinct().rdd.flatMap(lambda x: x).collect()
    random_pairs = random.sample(user_ids, 200)
    random_pairs = [(random_pairs[i], random_pairs[i + 1]) for i in range(0, len(random_pairs), 2)]

    random_correlations = []
    for user1, user2 in random_pairs:
        ratings_user1 = ratings.filter(ratings.userId == user1).select('movieId', 'rating').alias('ratings1')
        ratings_user2 = ratings.filter(ratings.userId == user2).select('movieId', 'rating').alias('ratings2')
        joined_ratings = ratings_user1.join(ratings_user2, 'movieId')
        corr_value = joined_ratings.select(corr('ratings1.rating', 'ratings2.rating')).collect()[0][0]
        if corr_value is not None:
            random_correlations.append(corr_value)

    average_random_correlation = sum(random_correlations) / len(random_correlations)

    print("Average Correlation (Movie Twins):", average_correlation)
    print("Average Correlation (Random Pairs):", average_random_correlation)


# def train_val_test_split(df):
    filtered_unique_user = df.groupBy('userId').agg({'rating': 'count'}).withColumnRenamed('count(rating)', 'rating_count')
    filtered_unique_user = filtered_unique_user.filter(col('rating_count')>=10).select('userId')

    #     train_users = filtered_unique_user.sample(withReplacement=False, fraction=0.7, seed=2024)
    #     val_test_users = filtered_unique_user.select('userId').subtract(train_users.select('userId'))
    filtered_unique_user = filtered_unique_user.withColumn('rand', rand(seed=2024))
    filtered_unique_user = filtered_unique_user.orderBy('rand')
    filtered_unique_user = filtered_unique_user.drop(col('rand'))

    split_point = int(filtered_unique_user.count() * 0.7)

    train_users = filtered_unique_user.limit(split_point)
    val_test_users = filtered_unique_user.subtract(train_users)

    #     val_users = val_test_users.sample(withReplacement=False, fraction=0.5, seed=2024)
    #     test_users = val_test_users.select('userId').subtract(val_users.select('userId'))

    val_test_users = val_test_users.withColumn('rand', rand(seed=2024))
    val_test_users = val_test_users.orderBy('rand')
    val_test_users = val_test_users.drop(col('rand'))

    split_point = int(val_test_users.count() * 0.5)

    val_users = val_test_users.limit(split_point)
    test_users = val_test_users.subtract(val_users)

    print(train_users.count())
    print(val_test_users.count())
    print(val_users.count())
    print(test_users.count())

    train_ratings = df.join(train_users, on='userId', how='inner')
    val_ratings = df.join(val_users, on='userId', how='inner')
    test_ratings = df.join(test_users, on='userId', how='inner')

    val_users_list = [row["userId"] for row in val_users.collect()]
    test_users_list = [row["userId"] for row in test_users.collect()]

    val_fractions = dict(zip(val_users_list, [0.5 for _ in range(len(val_users_list))]))
    test_fractions = dict(zip(test_users_list, [0.5 for _ in range(len(test_users_list))]))
        
    val_train = val_ratings.sampleBy('userId', fractions=val_fractions, seed=2024)
    validation = val_ratings.subtract(val_train)

    test_train = test_ratings.sampleBy('userId', fractions=test_fractions, seed=2024)
    test = test_ratings.subtract(test_train)

    train = train_ratings.union(val_train).union(test_train)

    return train, validation, test


if __name__ == "__main__":
    # Create the spark session object
    spark = SparkSession.builder \
    .appName("Spark Application") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.sql.broadcastTimeout", "7200") \
    .config("spark.driver.extraJavaOptions", "-XX:+UseG1GC") \
    .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC") \
    .getOrCreate()
    # spark.sparkContext.setLogLevel("DEBUG")

    # Get user userID from the command line
    # We need this to access the user's folder in HDFS
    userID = os.environ['USER']

    # Call our main routine
    main(spark, userID)
