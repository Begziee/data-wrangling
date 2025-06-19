# utils.py
# This module contains utility functions for reading and cleaning data in a Spark DataFrame.
# It includes functions to read CSV files, clean address-related columns, and handle specific data formats.
# It is designed to be used in a Spark environment for data processing tasks.
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    when,
    lower,
    regexp_replace,
    coalesce,
    lit,
    regexp_extract,
    concat_ws,
    trim,
)
import os

if "spark" in globals():
    spark.stop()

spark = (
    SparkSession.builder.appName("HalperFunctions")
    .master("local[*]")
    .config("spark.driver.memory", "8g")
    .config("spark.executor.memory", "4g")
    .config("spark.memory.offHeap.enabled", "true")
    .config("spark.memory.offHeap.size", "2g")
    .getOrCreate()
)


def _read_csv(file_path):
    """
    Reads a CSV file into a Spark DataFrame, ensuring the file exists and is not empty.
    :param file_path: Path to the CSV file to be read.
    :return: Spark DataFrame containing the data from the CSV file.
    :raises ValueError: If the file path does not point to a CSV file or if the file is empty.
    :raises FileNotFoundError: If the file does not exist or is not accessible.
    """
    if not file_path.endswith(".csv"):
        raise ValueError("The provided file path must point to a CSV file.")
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file {file_path} does not exist or is not accessible."
        )

    print(f"Reading CSV file from: {file_path}")
    data = spark.sparkContext.textFile(file_path)
    if data.isEmpty():
        raise ValueError(f"The file {file_path} is empty.")
    # Read the CSV file into a DataFrame
    print(f"File {file_path} exists and is not empty.")
    data_rdd = data.zipWithIndex().filter(lambda x: x[1] >= 6).map(lambda x: x[0])
    data_csv = spark.read.options(header=True, inferSchema=True).csv(data_rdd)
    print(f"{data_csv.count()} records read from {file_path}")
    print(f"Deduplicating records ...")
    # Remove records with null or NaN in the 'id' column
    data_csv = data_csv.dropna(subset=["id"])
    data_csv = data_csv.filter(col("id").rlike("^[0-9]+$"))

    # This can be exposed to further investigate the duplicate records later in the future
    duplicate_ids = data_csv.groupBy("id").count().filter(col("count") > 1).select("id")
    duplicate_records = data_csv.join(duplicate_ids, on="id", how="inner")
    print(f"Found {duplicate_records.count()} duplicate records based on 'id' column.")

    # Clean records with duplicate IDs
    data_csv = data_csv.exceptAll(duplicate_records)
    print(f"{data_csv.count()} records after deduplication.")

    return data_csv


def _clean_effstart(df):
    """
    Cleans the 'effstart' column of a DataFrame by ensuring it follows the format 'dd/mm/yyyy'.
    If the format is invalid, it replaces 'effstart' with 'effend', 'effend' with 'streetadd',
    and sets 'streetadd' to None based on business logic.
    :param df: Spark DataFrame whose 'effstart' column needs to be cleaned.
    :return: DataFrame with cleaned 'effstart' column.
    """
    expected_regex = "^\d{2}/\d{2}/\d{4}$"
    invalid_effstart_df = df.filter(~col("effstart").rlike(expected_regex))

    print("Cleaning column 'effstart' ...")
    print(
        f"Found {invalid_effstart_df.count()} records with invalid 'effstart' format."
    )

    if invalid_effstart_df.count() > 0:
        print("Replacing invalid 'effstart' based on business logic...")
        df = (
            df.withColumn(
                "effstart",
                when(~col("effstart").rlike(expected_regex), col("effend")).otherwise(
                    col("effstart")
                ),
            )
            .withColumn(
                "effend",
                when(
                    ~col("effstart").rlike(expected_regex), col("streetadd")
                ).otherwise(col("effend")),
            )
            .withColumn(
                "streetadd",
                when(~col("effstart").rlike(expected_regex), lit(None)).otherwise(
                    col("streetadd")
                ),
            )
        )
        print(f"Completed")
    else:
        print("All records have valid 'effstart' format.")

    return df


def _clean_post_code(df):
    """
    Cleans the 'postal_code' column of a DataFrame by normalizing city and state names,
    and applying business logic to determine the correct country and postal code.
    The function checks if the postal code is a valid 4 or 5 digit number,
    or if it matches specific patterns, and updates the 'country_new' and 'postalcode_new' columns accordingly.
    :param df: Spark DataFrame whose 'postal_code' column needs to be cleaned.
    :return: DataFrame with cleaned 'postal_code' column and new columns 'country_new' and 'postalcode_new'.
    """
    print("Cleaning 'post_code' column ...")
    city_norm = lower(regexp_replace(col("city"), "[áéíóúüñ]", "a"))
    state_norm = lower(regexp_replace(col("state"), "[áéíóúüñ]", "a"))

    invalid_postal_code_df = df.filter(~col("postal_code").rlike("^[0-9]{4,5}$"))
    print(
        f"Found {invalid_postal_code_df.count()} records with invalid 'postal_code' format."
    )
    if invalid_postal_code_df.count() > 0:
        print("Applying business logic to clean 'postal_code' ...")

        df = df.withColumn(
            "country_new",
            when(
                col("postal_code").rlike("^[0-9]{4,5}$") | col("postal_code").isNull(),
                col("country"),
            ).otherwise(
                when(city_norm == state_norm, col("country"))
                .when(
                    ~col("postal_code").rlike("^\d{2}/\d{2}/\d{4}$"), col("postal_code")
                )
                .otherwise(col("state"))
            ),
        )
        df = df.withColumn(
            "postalcode_new",
            when(
                col("postal_code").rlike("^[0-9]{4,5}$"), col("postal_code")
            ).otherwise(
                coalesce(
                    when(col("city").rlike("^[0-9]{4,5}$"), col("city")),
                    when(col("state").rlike("^[0-9]{4,5}$"), col("state")),
                    when(col("country").rlike("^[0-9]{4,5}$"), col("country")),
                    when(col("effstart").rlike("^[0-9]{4,5}$"), col("effstart")),
                    when(col("effend").rlike("^[0-9]{4,5}$"), col("effend")),
                    when(col("streetadd").rlike("^[0-9]{4,5}$"), col("streetadd")),
                )
            ),
        )
        print(f"Completed")
    else:
        print("All records have valid 'postal_code' format.")

    return df


def _clean_customerid(df):
    """
    Cleans the 'customerid' column of a DataFrame by ensuring it follows the format 'XX-12345'.
    If the format is invalid, it replaces 'customerid' with a stripped version based on business logic,
    and updates 'city' and 'streetadd' columns accordingly.
    :param df: Spark DataFrame whose 'customerid' column needs to be cleaned.
    :return: DataFrame with cleaned 'customerid' column and updated 'city' and 'streetadd' columns.
    """
    expected_regex = "^[A-Za-z]{2}-\d+$"
    invalid_customerid_df = df.filter(~col("customerid").rlike(expected_regex))

    print("Cleaning column 'customerid' ...")
    print(
        f"Found {invalid_customerid_df.count()} records with invalid 'effstart' format."
    )

    if invalid_customerid_df.count() > 0:
        print("Replacing invalid 'effstart' based on business logic...")
        df = df.withColumn(
            "customerid_stripped",
            regexp_extract(col("customerid"), r"^[A-Za-z]{2}-\d+(.*)$", 1),
        )
        df = df.withColumn(
            "city",
            when(
                col("city").isNull() & (col("customerid_stripped") != ""),
                col("customerid_stripped"),
            ).otherwise(col("city")),
        )
        df = df.withColumn(
            "streetadd",
            when(
                ~col("effend").rlike("^\d{2}/\d{2}/\d{4}$")
                & (col("customerid_stripped") != ""),
                concat_ws(" ", col("effend"), col("customerid_stripped")),
            ).otherwise(col("streetadd")),
        )
        df = df.withColumn(
            "customerid", regexp_extract(col("customerid"), r"([A-Za-z]{2}-\d+)", 1)
        )
        df = df.drop("customerid_stripped")
        print(f"Completed")
    else:
        print("All records have valid 'effstart' format.")

    return df


def _clean_effend(df):
    """
    Cleans the 'effend' column of a DataFrame by ensuring it follows the format 'dd/mm/yyyy'.
    If the format is invalid, it replaces 'effend' with None based on business logic.
    :param df: Spark DataFrame whose 'effend' column needs to be cleaned.
    :return: DataFrame with cleaned 'effend' column.
    """
    expected_regex = "^\d{2}/\d{2}/\d{4}$"
    invalid_effend_df = df.filter(~col("effend").rlike(expected_regex))

    print("Cleaning column 'effend' ...")
    print(f"Found {invalid_effend_df.count()} records with invalid 'effend' format.")

    if invalid_effend_df.count() > 0:
        print("Replacing invalid 'effend' based on business logic...")
        df = df.withColumn(
            "effend",
            when(~col("effend").rlike(expected_regex), lit(None)).otherwise(
                col("effend")
            ),
        )
        print(f"Completed")
    else:
        print("All records have valid 'effend' format.")

    return df


def _clean_country(df):
    """
    Cleans the 'country' column of a DataFrame by ensuring it follows the format 'XX'.
    If the format is invalid, it replaces 'country' with None based on business logic.
    :param df: Spark DataFrame whose 'country' column needs to be cleaned.
    :return: DataFrame with cleaned 'country' column.
    """

    print("Cleaning column 'country' ...")

    df = df.withColumn(
        "country_new", trim(regexp_replace(col("country_new"), r"^[^A-Za-z]*", ""))
    )

    df = df.withColumn(
        "country_new",
        when(
            col("country_new").isin("US", "USA", "United States"),
            "United States of America",
        )
        .when(col("country_new") == "Alger", "Algeria")
        .when(
            col("country_new").isin("Republic of the Congo", "Congo"),
            "Democratic Republic of the Congo",
        )
        .when(col("country_new") == "NZ", "New Zealand")
        .when(col("country_new") == "UK", "United Kingdom")
        .otherwise(col("country_new")),
    )
    print(f"Completed")

    return df


def _select_columns(df):
    """
    Selects specific columns from the DataFrame and renames them for consistency.
    :param df: Spark DataFrame to select columns from.
    :return: DataFrame with selected and renamed columns.
    """
    print("Selecting and renaming columns ...")
    df = df.select(
        col("id").alias("id"),
        col("customerid").alias("customer_id"),
        col("effstart").alias("effective_start_date"),
        col("effend").alias("effective_end_date"),
        col("streetadd").alias("street_address"),
        col("city").alias("city"),
        col("state").alias("state"),
        col("postal_code").alias("postal_code"),
        col("country_new").alias("country"),
        col("postalcode_new").alias("postal_code"),
    )
    print(f"Completed")
    return df


def _clean_address(df):
    """
    Cleans the address-related columns in the DataFrame by applying various cleaning functions.
    This includes cleaning postal codes, effective start and end dates, customer IDs, and country names.
    :param df: Spark DataFrame containing address-related columns to be cleaned.
    :return: DataFrame with cleaned address-related columns.
    """

    print("Cleaning address columns ...")

    df1 = _clean_post_code(df)
    df2 = _clean_effstart(df1)
    df3 = _clean_customerid(df2)
    df4 = _clean_effend(df3)
    df5 = _clean_country(df4)
    df6 = _select_columns(df5)

    print("Address columns cleaned successfully.")
    return df6
