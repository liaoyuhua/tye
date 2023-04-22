from typing import List, Dict, Optional
import logging
import subprocess
from copy import copy
import pandas as pd
from torch.utils.data import Dataset
from tye.utils import tolist


logger = logging.getLogger(__name__)


def sql2csv(table: str, file_path: str):
    """
    Args:
        table: The table name in database.
        file_path: The file path to save the dataset.
    """
    base_shell = """
    
    hive -e "
    
    SELECT * FROM {table}

    " > {file_path}
    """
    sql = base_shell.format(table=table, file_path=file_path)

    subprocess.check_call(sql)


class DataCreator:
    def __init__(
        self,
        df: pd.DataFrame,
        cols: List[str],
        embedding: pd.DataFrame,
        emb_col: str,
        parse_emb: bool = True,
    ):
        """
        Args:
            df: The dataset.
            cols: Columns to be replced by embedding. The columns should
            be the same as the embedding dataframe.
            embedding: The embedding dataframe.
            emb_col: The embedding column name.
            parse_emb: Whether to parse the embedding dataframe.
        """
        assert (
            cols is not None and embedding is not None
        ), "Please provide the embedding columns and the embedding dataframe."
        self.cols = tolist(cols)
        self.embedding = (
            embedding if not parse_emb else self._parse_emb(embedding, emb_col)
        )
        self.df = self._build(df, self.embedding, self.cols, emb_col)

    def _build(
        self,
        df: pd.DataFrame,
        embedding: pd.DataFrame,
        cols: List[str],
    ):
        """
        This function is used to build the dataset.

        Args:
            df: The dataset.
            embedding: The embedding dataframe.
            cols: Columns to be replced by embedding.
            emb_col: The embedding column name.

        Returns:
            The built dataset.
        """
        df = df.copy()
        len_before = len(df)
        df = df.merge(embedding, left_on=cols, right_on=cols, how="inner")
        len_after = len(df)

        if len_before != len_after:
            logger.warning(
                "The length of the dataset before and after building is different."
            )
            logger.warning("The length before: {}".format(len_before))
            logger.warning("The length after: {}".format(len_after))
            logger.warning("The difference: {}".format(len_before - len_after))

        return df

    @classmethod
    def from_sql(
        cls,
        tbl_name: str,
        file_path: str,
        cols: List[str] = None,
        embedding: pd.DataFrame = None,
        emb_col: str = None,
        parse_emb: bool = True,
    ):
        df = sql2csv(tbl_name, file_path)
        logger.info("Successfully load the dataset from sql script.")
        return cls(df, cols, embedding, emb_col, parse_emb)

    @classmethod
    def from_csv(
        cls,
        file_path: str,
        sep: Optional[str] = ",",
        header: Optional[List[str]] = None,
        index_col: Optional[str] = None,
        cols: List[str] = None,
        embedding: pd.DataFrame = None,
        emb_col: str = None,
        parse_emb: bool = True,
    ):
        df = pd.read_csv(file_path, sep=sep, header=header, index_col=index_col)
        logger.info("Successfully load the dataset from csv file.")
        return cls(df, cols, embedding, emb_col, parse_emb)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        cols: List[str] = None,
        embedding: pd.DataFrame = None,
        emb_col: str = None,
        parse_emb: bool = True,
    ):
        logger.info("Successfully load the dataset from dataframe.")
        return cls(df, cols, embedding, emb_col, parse_emb)

    def _parse_emb(self, embedding: pd.DataFrame, emb_col: str):
        """
        This function is used to parse the embedding dataframe.

        For example, the embedding dataframe is like:
        +---------+---------+-----------------+
        | source  | target  | embedding       |
        +---------+---------+-----------------+
        | A       | B       | [0.1, 0.2, -0.1]|
        +---------+---------+-----------------+
        | C       | D       | [0.2, 0.3, -0.2]|
        +---------+---------+-----------------+

        Then, the parsed embedding dataframe is like:
        +---------+---------+-----------------+-----------------+-----------------+
        | source  | target  | embedding_0     | embedding_1     | embedding_2     |
        +---------+---------+-----------------+-----------------+-----------------+
        | A       | B       | 0.1             | 0.2             | -0.1            |
        +---------+---------+-----------------+-----------------+-----------------+
        | C       | D       | 0.2             | 0.3             | -0.2            |
        +---------+---------+-----------------+-----------------+-----------------+

        Args:
            embedding: The embedding dataframe.
            emb_col: The embedding column name.

        Returns:
            The parsed embedding dataframe.
        """
        emb = embedding.copy()
        emb[emb_col] = emb[emb_col].apply(lambda x: eval(x))
        emb = pd.DataFrame(
            emb[emb_col].tolist(),
            index=emb.index,
            columns=[emb_col + "_" + str(i) for i in range(len(emb[emb_col][0]))],
        )
        emb = pd.concat([embedding, emb], axis=1).drop(emb_col, axis=1, inplace=True)

        return emb


class TSDataset(Dataset):
    """
    Time series Dataset for training and evaluating the performance of gnn embedding.

    Note that all time series in the dataset must have the same length.

    For example, the dataset is like:
    +------+-----------------+-----------------+-----------------+-------+
    | time | embedding_0     | embedding_1     | embedding_2     | value |
    +------+-----------------+-----------------+-----------------+-------+
    | 1    | 0.1             | 0.2             | -0.1            | 0.1   |
    +------+-----------------+-----------------+-----------------+-------+
    | 2    | 0.2             | 0.3             | -0.2            | 0.2   |
    +------+-----------------+-----------------+-----------------+-------+
    | 3    | 0.3             | 0.4             | -0.3            | 0.3   |
    +------+-----------------+-----------------+-----------------+-------+
    """

    def __init__(
        self,
        data: pd.DataFrame,
        time_col: str,
        value_col: str,
        embedding_cols: List[str],
        train_ratio=0.8,
        val_ratio=0.1,
    ) -> None:
        super().__init__()
        self.time_col = time_col
        self.value_col = value_col
        self.embedding_cols = tolist(embedding_cols)

        self.data = data.sort_values(
            by=[self.time_col] + self.embedding_cols
        ).reset_index(drop=True)

        self._check()

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

        self.train, self.val, self.test = self._split()

    def _check(self):
        """
        This function is used to check whether all time series in dataframe have the same size.
        """

        def check_time_series(df):
            if len(df) != len(self.data):
                raise ValueError(
                    "The length of the time series is not the same as the length of the dataset."
                )

        self.data.groupby(self.embedding_cols).apply(check_time_series)

    def __len__(self):
        """
        The length of each time series.
        """
        return len(self.data) // len(self.data[self.embedding_cols].drop_duplicates())

    def _split(self):
        """
        Split the dataset into train, validation and test set.

        Args:
            train_ratio: The ratio of the train set.
            val_ratio: The ratio of the validation set.

        Returns:
            The train, validation and test set.
        """
        self._check()

        train_size = int(len(self) * self.train_ratio)
        val_size = int(len(self) * self.val_ratio)
        test_size = len(self) - train_size - val_size

        train_set = self.data[:train_size]
        val_set = self.data[train_size : train_size + val_size]
        test_set = self.data[train_size + val_size :]

        return train_set, val_set, test_set

    def __getitem__(self, idx):
        """
        Get the time series at index idx.

        Args:
            idx: The index of the time series.

        Returns:
            The time series at index idx.
        """
        return self.data[idx * len(self) : (idx + 1) * len(self)]
