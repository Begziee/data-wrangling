from pyspark.sql import SparkSession

from helper.utils import _read_txt

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

region_df = _read_txt("data/region.txt")
# shipping_address_df.show()
region_df.show()
