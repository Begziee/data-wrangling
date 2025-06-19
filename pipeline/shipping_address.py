from pyspark.sql import SparkSession

from helper.utils import _read_csv, _clean_address

if "spark" in globals():
    spark.stop()

spark = (
    SparkSession.builder.appName("ExplorationApp")
    .master("local[*]")
    .config("spark.sql.debug.maxToStringFields", "100")
    .config("spark.driver.memory", "8g")
    .config("spark.executor.memory", "4g")
    .config("spark.memory.offHeap.enabled", "true")
    .config("spark.jars.packages", "com.databricks:spark-xml_2.12:0.15.0")
    .config("spark.memory.offHeap.size", "2g")
    .getOrCreate()
)

shipping_address_df = _read_csv("data/shippuingaddress_20240521.csv.csv")
# shipping_address_df.show()
shipping_address_df = _clean_address(shipping_address_df)
shipping_address_df.show()
