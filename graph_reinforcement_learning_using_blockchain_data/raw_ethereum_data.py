import argparse
import asyncio
import os
import random
from argparse import ArgumentError
from typing import List, Coroutine, Any, Dict

import dask.dataframe as dd
import pandas as pd
from aiohttp import ClientResponseError
from loguru import logger
from tenacity import retry, wait_exponential, stop_after_attempt
from tqdm import tqdm
from web3 import AsyncWeb3, AsyncHTTPProvider
from web3.types import TxReceipt

from graph_reinforcement_learning_using_blockchain_data.config import (
    EXTERNAL_DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)

ALCHEMY_API_URL = os.getenv("ALCHEMY_API_URL")
random.seed(42)


async def gather_with_tqdm(coros: List[Coroutine], desc: str = "Fetching logs") -> List[Any]:
    """
    Execute coroutines concurrently with a progress bar.

    :param coros: List of coroutines to execute concurrently
    :param desc: Description for the progress bar
    :return: List of results from the coroutines in the same order
    """

    async def task_wrapper(coro, pbar):
        result = await coro
        pbar.update(1)
        return result

    with tqdm(total=len(coros), desc=desc) as pbar:
        wrapped = [task_wrapper(coro, pbar) for coro in coros]
        results = await asyncio.gather(*wrapped)
    return results


class Dataset:
    """
    Class for retrieving Ethereum blockchain data via a Web3 provider.

    Provides methods to fetch transaction receipts, ETH balances, and transactions per block.
    Uses asyncio and semaphores to manage concurrent API requests.
    """

    def __init__(self):
        """
        Initialize a Dataset instance with Web3 provider and request rate limiter.
        """
        self.web3 = AsyncWeb3(AsyncHTTPProvider(ALCHEMY_API_URL))
        self.sem = asyncio.Semaphore(5)

    def fetch_eth_balances(
        self, accounts: List[str], block_numbers: List[int]
    ) -> List[Dict[str, Any]]:
        """
        Fetch ETH balances for multiple accounts at specified block numbers.

        :param accounts: List of Ethereum account addresses
        :param block_numbers: List of block numbers corresponding to each account
        :return: List of dictionaries containing account, block_number, and balance
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        balances_list = loop.run_until_complete(
            self._fetch_eth_balances_async(accounts, block_numbers)
        )
        loop.close()
        return balances_list

    def fetch_logs_per_transaction(self, trxs: List[str]) -> List[TxReceipt]:
        """
        Fetch transaction receipts for a list of transaction hashes.

        :param trxs: List of transaction hashes to fetch receipts for
        :return: List of transaction receipts (TxReceipt objects)
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        tasks = [self._get_logs(tx) for tx in trxs]
        results = loop.run_until_complete(gather_with_tqdm(tasks, desc="Fetching logs"))
        loop.close()

        return results

    def fetch_transactions_per_block(self, block_number: int) -> List[str]:
        """
        Fetch all transaction hashes in a specific block.

        :param block_number: The block number to fetch transactions from
        :return: List of transaction hashes in hexadecimal format
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        block = loop.run_until_complete(self.web3.eth.get_block(block_number))
        loop.close()
        trxs = block.get("transactions", [])
        return [trx.hex() for trx in trxs]

    @retry(wait=wait_exponential(min=1, max=60), stop=stop_after_attempt(5))
    async def _get_logs(self, tx_hash: str) -> TxReceipt:
        """
        Get transaction receipt for a transaction hash with retry logic.

        :param tx_hash: Transaction hash to fetch receipt for
        :return: Transaction receipt object
        :raises: ClientResponseError if the request fails after retries
        """
        async with self.sem:
            try:
                receipt = await self.web3.eth.get_transaction_receipt(tx_hash)
                return receipt
            except ClientResponseError as e:
                if e.status == 429:
                    raise
                raise

    async def _fetch_eth_balances_async(
        self, accounts: List[str], block_numbers: List[int]
    ) -> List[Dict[str, Any]]:
        """
        Fetch ETH balances for multiple accounts asynchronously.

        :param accounts: List of Ethereum account addresses
        :param block_numbers: List of block numbers corresponding to each account
        :return: List of dictionaries containing account, block_number, and balance
        """
        unique_pairs = list({(acc, bn) for acc, bn in zip(accounts, block_numbers)})
        tasks = [self._get_eth_balance(acc, bn) for acc, bn in unique_pairs]
        return await gather_with_tqdm(tasks, desc="Fetching ETH balances")

    @retry(wait=wait_exponential(min=1, max=60), stop=stop_after_attempt(5))
    async def _get_eth_balance(self, account: str, block_number: int) -> Dict[str, Any]:
        """
        Get ETH balance for a specific account at a specific block with retry logic.

        :param account: Ethereum account address
        :param block_number: Block number to fetch balance at
        :return: Dictionary containing account, block_number, and balance
        :raises: ClientResponseError if the request fails after retries
        """
        async with self.sem:
            try:
                balance = await self.web3.eth.get_balance(account, block_identifier=block_number)
                return {"account": account, "block_number": block_number, "balance": balance}
            except ClientResponseError as e:
                if e.status == 429:
                    raise
                raise


def _subsample(ddf: dd.DataFrame, rows: int) -> dd.DataFrame:
    """
    Subsample a Dask DataFrame to a specified number of rows.

    :param ddf: Dask DataFrame to subsample
    :param rows: Number of rows to sample (-1 to keep all rows)
    :return: Subsampled Dask DataFrame
    """
    if rows == -1:
        return ddf
    ddf.sort_values(by="block_number", inplace=True)
    dd_subsample = ddf.head(rows)
    ddf = dd.from_pandas(dd_subsample, npartitions=4)
    return ddf


def _get_data_class_0(ds: Dataset, rows: int, filename: str) -> None:
    """
    Process and save class 0 data (non-arbitrage transactions).

    Samples normal transactions from the same blocks where arbitrage transactions
    occurred, excluding the arbitrage transactions themselves.

    :param ds: Dataset instance for fetching blockchain data
    :param rows: Number of rows to subsample from the arbitrage dataset (-1 for all)
    :param filename: Output filename for the processed data
    :return: None
    """
    ddf = dd.read_csv(
        EXTERNAL_DATA_DIR / "flashbots" / "Q2_2023" / "arbitrages.csv",
        dtype={
            "end_amount": "object",
            "error": "object",
            "profit_amount": "object",
            "start_amount": "object",
        },
    )
    ddf = _subsample(ddf, rows)
    flashbots_tx = set(ddf["transaction_hash"].compute().tolist())
    blocks = ddf["block_number"].compute().unique().tolist()

    block_numbers = []
    tx_hashes = []
    for block in tqdm(blocks, desc="Processsing Blocks"):
        all_txs = ds.fetch_transactions_per_block(block)
        max_normal_per_block = int(0.05 * len(all_txs))
        filtered_txs = [tx for tx in all_txs if tx not in flashbots_tx]
        if len(filtered_txs) > max_normal_per_block:
            filtered_txs = random.sample(filtered_txs, max_normal_per_block)
        for tx in filtered_txs:
            block_numbers.append(block)
            tx_hashes.append(tx)

    df_class_0 = pd.DataFrame({"block_number": block_numbers, "transaction_hash": tx_hashes})

    receipts = ds.fetch_logs_per_transaction(tx_hashes)
    df_class_0["receipt"] = receipts

    df_class_0.to_csv(PROCESSED_DATA_DIR / "flashbots" / "Q2_2023" / filename, index=False)


def _get_data_class_1(ds: Dataset, rows: int, filename: str) -> None:
    """
    Process and save class 1 data (arbitrage transactions).

    Fetches transaction receipts for arbitrage transactions from the flashbots dataset.

    :param ds: Dataset instance for fetching blockchain data
    :param rows: Number of rows to subsample from the arbitrage dataset (-1 for all)
    :param filename: Output filename for the processed data
    :return: None
    """
    ddf = dd.read_csv(
        EXTERNAL_DATA_DIR / "flashbots" / "Q2_2023" / "arbitrages.csv",
        dtype={
            "end_amount": "object",
            "error": "object",
            "profit_amount": "object",
            "start_amount": "object",
        },
    )
    ddf = _subsample(ddf, rows)
    trxs = ddf["transaction_hash"].compute().tolist()
    receipts = ds.fetch_logs_per_transaction(trxs)

    df = ddf.compute()
    df["receipt"] = receipts
    df.to_csv(PROCESSED_DATA_DIR / "flashbots" / "Q2_2023" / filename, index=False)


def _get_eth_balances(ds: Dataset, input_file: str, output_file: str) -> None:
    """
    Fetch and save ETH balances for accounts in a given file.

    :param ds: Dataset instance for fetching blockchain data
    :param input_file: Path to input CSV file containing account addresses and block numbers
    :param output_file: Path to output CSV file to save ETH balances
    :return: None
    """
    df = pd.read_csv(RAW_DATA_DIR / input_file)
    eth_balances = ds.fetch_eth_balances(df["from"].tolist(), df["blockNumber"].tolist())
    df_balances = pd.DataFrame(eth_balances)
    df_balances.to_csv(RAW_DATA_DIR / output_file, index=False)


def main() -> None:
    """
    Main function to parse command line arguments and execute the appropriate data processing tasks.

    Command line arguments:
    - data_class: Type of data to process (receipts0, receipts1, eth_balances0, eth_balances1)
    - rows: Number of rows to sample (-1 for all)
    - output_filename: Name of the output file
    - input_filename: Name of the input file (for ETH balance tasks)

    :return: None
    :raises ArgumentError: If invalid arguments are provided
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_class",
        choices=["receipts0", "receipts1", "eth_balances0", "eth_balances1"],
        default=None,
    )
    args0 = parser.parse_args()

    match args0:
        case "0" | "1":
            parser.add_argument("--rows", default=-1, type=int)
            parser.add_argument("--output_filename")
        case "eth_balances0":
            parser.add_argument("--input_filename", default="receipts_class0.csv")
            parser.add_argument("--output_filename", default="eth_balances_class0.csv")
        case "eth_balances1":
            parser.add_argument("--input_filename", default="receipts_class1.csv")
            parser.add_argument("--output_filename", default="eth_balances_class1.csv")
        case _:
            raise ArgumentError

    args = parser.parse_args()
    assert args.data_class, "Please provide a data class to process"

    ds = Dataset()

    logger.info("Processing dataset...")

    match args.data_class:
        case "0":
            _get_data_class_0(ds, int(args.rows), args.output_filename)
        case "1":
            _get_data_class_1(ds, int(args.rows), args.output_filename)
        case "eth_balances":
            logger.info(
                f"Will get ETH balances for accounts in {args.input_filename} and save as {args.output_filename}"
            )
            _get_eth_balances(ds, args.input_filename, args.output_filename)

    logger.success("Processing dataset complete.")


if __name__ == "__main__":
    main()
