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
    ratings = spark.read.csv(f'hdfs:/user/{userID}/ml-latest/ratings.csv', header = True, schema='userId INT, movieId string, rating FLOAT, timestamp INT')
    # movies = spark.read.csv(f'hdfs:/user/{userID}/ml-latest/movies.csv')
    # links = spark.read.csv(f'hdfs:/user/{userID}/ml-latest/links.csv')
    # tags = spark.read.csv(f'hdfs:/user/{userID}/ml-latest/tags.csv')
    
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
