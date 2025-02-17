import argparse
import asyncio
import os
import random

import dask.dataframe as dd
import pandas as pd
from aiohttp import ClientResponseError
from loguru import logger
from tenacity import retry, wait_exponential, stop_after_attempt
from tqdm import tqdm
from web3 import AsyncWeb3, AsyncHTTPProvider

from graph_reinforcement_learning_using_blockchain_data.config import (
    EXTERNAL_DATA_DIR,
    PROCESSED_DATA_DIR,
)

ALCHEMY_API_URL = os.getenv("ALCHEMY_API_URL")
random.seed(42)


async def gather_with_tqdm(coros, desc="Fetching logs"):
    results = []
    for coro in tqdm(asyncio.as_completed(coros), total=len(coros), desc=desc):
        results.append(await coro)
    return results


class Dataset:
    def __init__(self):
        self.web3 = AsyncWeb3(AsyncHTTPProvider(ALCHEMY_API_URL))
        self.sem = None

    def fetch_logs_per_transaction(self, trxs: list) -> dd.DataFrame:
        """
        args:
            trxs: list of transaction hashes to fetch logs for
        returns:
            list of logs for each transaction
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        self.sem = asyncio.Semaphore(5)

        tasks = [self._get_logs(tx) for tx in trxs]
        results = loop.run_until_complete(gather_with_tqdm(tasks, desc="Fetching logs"))
        loop.close()

        return results

    def fetch_transactions_per_block(self, block_number: int) -> list:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        block = loop.run_until_complete(self.web3.eth.get_block(block_number))
        loop.close()
        trxs = block.get("transactions", [])
        return [trx.hex() for trx in trxs]

    @retry(wait=wait_exponential(min=1, max=60), stop=stop_after_attempt(5))
    async def _get_logs(self, tx_hash: str) -> list:
        async with self.sem:
            try:
                receipt = await self.web3.eth.wait_for_transaction_receipt(tx_hash)
                return receipt
            except ClientResponseError as e:
                if e.status == 429:
                    raise
                raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_class", choices=["0", "1"], default=None)

    parser.add_argument("--rows", default=-1, type=int)

    parser.add_argument("--filename")

    args = parser.parse_args()
    assert args.data_class, "Please provide a data class to process"

    ds = Dataset()

    def _subsample(ddf: dd.DataFrame, rows: int) -> dd.DataFrame:
        if rows == -1:
            return ddf
        ddf.sort_values(by="block_number", inplace=True)
        dd_subsample = ddf.head(rows)
        ddf = dd.from_pandas(dd_subsample, npartitions=4)
        return ddf

    def _get_data_class_0(rows: int, filename: str):
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
            # if len(filtered_txs) > max_normal_per_block:
            #     filtered_txs = random.sample(filtered_txs, max_normal_per_block)
            for tx in filtered_txs:
                block_numbers.append(block)
                tx_hashes.append(tx)

        df_class_0 = pd.DataFrame({"block_number": block_numbers, "transaction_hash": tx_hashes})

        receipts = ds.fetch_logs_per_transaction(tx_hashes)
        df_class_0["receipt"] = receipts

        df_class_0.to_csv(PROCESSED_DATA_DIR / "flashbots" / "Q2_2023" / filename, index=False)

    def _get_data_class_1(rows: int, filename: str):
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

    logger.info("Processing dataset...")

    if args.data_class == "0":
        _get_data_class_0(int(args.rows), args.filename)
    elif args.data_class == "1":
        _get_data_class_1(int(args.rows), args.filename)

    logger.success("Processing dataset complete.")


if __name__ == "__main__":
    main()
