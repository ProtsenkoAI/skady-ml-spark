from pyspark.sql import SparkSession
from pyspark.streaming import DStream, StreamingContext

from .consts import row_schema, Types
from util import read_config
from ..data_proc.fit_preproc import FitDataPreprocessor
from ..data_proc.data_managers import SparkStreamingFitDataManager
config = read_config()

# TODO: move consts to consts/
SomeModelType = None


def start_auto_fitting():
    # TODO: create fit_data_preprocessor here and use it
    data_stream, session = create_dstream(config, return_session=True)
    data_processor = FitDataPreprocessor(SparkStreamingFitDataManager())

    update_users_history(data_stream)
    fit_interacts = data_processor.create_fit_data(data_stream)
    start_receiving_results(fit_interacts)
    # model = get_or_load_model()
    # fit_model(fit_interacts, model)
    # save_model(model)


def create_dstream(config, return_session=False) -> Types.RawData:
    # master is initialized automatically (in spark-submit mode it's taken from --master argument
    spark = SparkSession\
        .builder\
        .appName(config["app_name"])\
        .getOrCreate()

    stream_type = config["data_stream"]["type"]
    if stream_type == "text_files_dir":
        stream_reader = spark.readStream
        data_stream = stream_reader.csv(config["data_stream"]["path"], schema=row_schema)
        # data_stream = StreamingContext(spark.sparkContext, 1).textFileStream(config["data_stream"]["path"])
    else:
        raise ValueError(f"The type of data_stream is not supported: {stream_type}")
    if return_session:
        return data_stream, spark
    return data_stream


def start_receiving_results(stream_df: Types.FitData):
    # TODO: test it
    def print_df(df, batch_id):
        print("Batch processed:", df, batch_id)
        print(df.collect())

    streaming_query = stream_df.writeStream.outputMode("append").foreachBatch(print_df).start()
    streaming_query.awaitTermination()


def update_users_history(interacts: Types.RawData):
    """Saves interacts to long-term history to load use it later"""
    # TODO
    pass


def get_or_load_model() -> SomeModelType:
    raise NotImplementedError


def fit_model(inters, model: SomeModelType):
    raise NotImplementedError


def save_model(model: SomeModelType):
    raise NotImplementedError
