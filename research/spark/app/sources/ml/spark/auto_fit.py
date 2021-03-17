from pyspark.sql import SparkSession
from pyspark.streaming import DStream, StreamingContext

import os
from .consts import row_schema, Types
from util import read_config
from ..data_proc.fit_preproc import FitDataPreprocessor
from ..data_proc.data_managers import SparkStreamingFitDataManager
from ..modeling.fitters import SparkFitter
from ..modeling.models import MFWithBiasModel
config = read_config()

# TODO: kkep only last batch in create_fit_data when using streaming
# TODO: move consts to consts/
SomeModelType = None


def start_auto_fitting():
    # TODO: create fit_data_preprocessor here and use it
    data_stream, session = create_dstream(config, return_session=True)
    # TODO: move setting model params to another place
    model = MFWithBiasModel(100, 100, 10)
    fitter = SparkFitter(config["train_params"], model)
    data_processor = FitDataPreprocessor(SparkStreamingFitDataManager())

    fit_interacts = data_processor.create_fit_data(data_stream)
    fitter.fit_model(fit_interacts)


def create_dstream(config, return_session=False) -> Types.RawData:
    # master is initialized automatically (in spark-submit mode it's taken from --master argument
    spark = SparkSession\
        .builder\
        .appName(config["app_name"])\
        .getOrCreate()

    stream_type = config["fit_stream"]["type"]
    stream_path = os.path.join(config["base_path"], config["fit_stream"]["relative_path"])
    if stream_type == "csv":
        stream_reader = spark.readStream
        data_stream = stream_reader.csv(stream_path, schema=row_schema)
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


def get_or_load_model() -> SomeModelType:
    raise NotImplementedError


def fit_model(inters, model: SomeModelType):
    raise NotImplementedError


def save_model(model: SomeModelType):
    raise NotImplementedError
