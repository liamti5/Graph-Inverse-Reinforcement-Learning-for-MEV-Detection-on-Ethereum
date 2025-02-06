import asyncio
import os

import dask.dataframe as dd
import pandas as pd
import typer
from aiohttp import ClientResponseError
from dask.diagnostics import ProgressBar
from loguru import logger
from tenacity import retry, wait_exponential, stop_after_attempt
from web3 import AsyncWeb3, AsyncHTTPProvider

from graph_reinforcement_learning_using_blockchain_data.config import (
    EXTERNAL_DATA_DIR,
    PROCESSED_DATA_DIR,
)

ALCHEMY_API_URL = os.getenv("ALCHEMY_API_URL")
web3 = AsyncWeb3(AsyncHTTPProvider(ALCHEMY_API_URL))

app = typer.Typer()

def _enhance_flashbots_arbs(data: dd.DataFrame) -> dd.DataFrame:
    """
    Called once per Dask partition.
    Creates a local event loop and local semaphore to avoid
    'Semaphore object is bound to a different event loop' errors.
    """
    data = data.copy()

    @retry(wait=wait_exponential(min=1, max=60), stop=stop_after_attempt(5))
    async def _get_logs(tx_hash: str) -> list:
        async with sem:
            try:
                receipt = await web3.eth.wait_for_transaction_receipt(tx_hash)
                return receipt.get("logs", [])
            except ClientResponseError as e:
                if e.status == 429:
                    raise
                raise

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    sem = asyncio.Semaphore(5)

    tasks = [_get_logs(tx) for tx in data["transaction_hash"]]
    results = loop.run_until_complete(asyncio.gather(*tasks))
    loop.close()

    data["logs"] = results
    return data


@app.command()
def main():
    logger.info("Processing dataset...")

    df = dd.read_csv(
        EXTERNAL_DATA_DIR / "flashbots" / "Q2_2023" / "arbitrages.csv",
        dtype={
            "end_amount": "object",
            "error": "object",
            "profit_amount": "object",
            "start_amount": "object",
        }
    )
    # df_subsample = df.head(100)
    # df = dd.from_pandas(df_subsample, npartitions=4)
    df = df.repartition(npartitions=4)
    meta = df._meta.copy()
    meta["logs"] = pd.Series(dtype=object)

    with ProgressBar():
        df = df.map_partitions(_enhance_flashbots_arbs, meta=meta)
        df = df.compute()

    df.to_csv(PROCESSED_DATA_DIR / "flashbots" / "Q2_2023" / "arbitrages_with_logs.csv", index=False)

    logger.success("Processing dataset complete.")


if __name__ == "__main__":
    app()
