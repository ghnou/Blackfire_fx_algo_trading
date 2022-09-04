import pyarrow.parquet as pq


def read_data(path, fx_pair, date_range):

    df = pq.read_table(
        source=path,
        use_threads=True,
        filters=[('YEAR', 'in', date_range), ('FX_PAIR', 'in', fx_pair)]
    )

    return df.to_pandas()


def save_data(df, path, partition_cols):

    df.to_parquet(
        path=path,
        engine='pyarrow',
        compression='snappy',
        partition_cols=partition_cols
    )